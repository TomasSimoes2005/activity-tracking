import os
import time
import random
import json
import threading
import pandas as pd
import yt_dlp
from concurrent.futures import ThreadPoolExecutor

# Selected AVA action classes for the recognition pipeline:
TARGET_CLASSES = [5, 8, 11, 12, 27, 29, 54, 57]

# Concurrency level (kept low to avoid YouTube IP rate-limiting):
MAX_WORKERS = 4

# Valid video container formats downloaded by yt-dlp:
VALID_EXTS = ('.mp4', '.webm', '.mkv', '.avi')

# Thread-safe lock to prevent workers from corrupting the JSON file when writing simultaneously:
blacklist_lock = threading.Lock()


def init_blacklist(output_dir="input/ava_kinetics/videos", blacklist_path="dead_links.json", reset=False):
    """
    Initializes the blacklist JSON ledger and scans local storage for already downloaded videos.
    :param output_dir: destination directory where downloaded video files are stored.
    :param blacklist_path: path to the JSON ledger file tracking dead links.
    :param reset: boolean flag indicating if the existing blacklist ledger should be wiped.
    :return: set of all treated video ID strings (both downloaded and blacklisted).
    """

    # Handle JSON blacklist initialization:
    dead_ids = set()
    if not os.path.exists(blacklist_path) or reset:
        with open(blacklist_path, 'w') as f:
            json.dump([], f, indent=4)
        print(f"[Init] Initialized clean blacklist ledger -> {blacklist_path}")
    else:
        try:
            with open(blacklist_path, 'r') as f:
                dead_ids = set(json.load(f))
            print(f"[Init] Loaded {len(dead_ids)} permanently dead links from -> {blacklist_path}")
        except Exception:
            print(f"[!] Warning: {blacklist_path} was corrupted. Resetting to empty list...")
            with open(blacklist_path, 'w') as f:
                json.dump([], f, indent=4)

    # Scan disk for files:
    os.makedirs(output_dir, exist_ok=True)
    existing_files = os.listdir(output_dir)
    disk_ids = {os.path.splitext(f)[0] for f in existing_files if f.endswith(VALID_EXTS)}
    print(f"[Init] Scanned disk and found {len(disk_ids)} verified video files in -> {output_dir}")

    # Merge into single set:
    treated_ids = disk_ids | dead_ids
    print(f"[Init] Total Treated IDs assembled in RAM (will be instantly skipped): {len(treated_ids)}\n")

    return treated_ids, disk_ids

def add_to_blacklist(vid, blacklist_path="dead_links.json"):
    """
    Safely appends a permanently broken video ID to the blacklist ledger using a thread lock.
    :param vid: unique YouTube video ID string to blacklist.
    :param blacklist_path: path to the JSON ledger file.
    """

    with blacklist_lock:
        dead_set = set()

        # Load existing IDs if file exists:
        if os.path.exists(blacklist_path):
            try:
                with open(blacklist_path, 'r') as f:
                    dead_set = set(json.load(f))
            except Exception:
                pass

        # Add new dead ID:
        dead_set.add(vid)

        # Write updated list back to file:
        with open(blacklist_path, 'w') as f:
            json.dump(list(dead_set), f, indent=4)


def _download_class_worker(action_id, class_vids, treated_ids, verified_count, target, output_dir, ydl_opts):
    """
    Worker function executed by each thread. Skips any ID in treated_ids and downloads until the disk quota is met.
    :param action_id: integer ID of the AVA action class assigned to this thread.
    :param class_vids: list of unique YouTube video ID strings associated with this action class.
    :param treated_ids: set of all treated video IDs (both downloaded and blacklisted) to skip in RAM.
    :param verified_count: integer starting count of physical files already verified on disk for this class.
    :param target: maximum number of valid video downloads required for this category.
    :param output_dir: destination directory to save downloaded video files.
    :param ydl_opts: configuration dictionary for yt-dlp downloader.
    :return: integer count of videos successfully downloaded or verified on disk for this class.
    """

    # If category is already completed from previous runs, skip thread:
    if verified_count >= target:
        print(f"[Class {action_id:02d}] Already COMPLETED ({verified_count}/{target} on disk). Skipping thread.")
        return verified_count

    print(f"[Worker -> Class {action_id:02d}] Starting! Currently verified: {verified_count}/{target}. Scanning queue...")

    # Scan through assigned category videos:
    for vid in class_vids:
        if verified_count >= target:
            break

        # Skip if it's in the blacklist:
        if vid in treated_ids:
            continue

        # Check all valid container formats to see if file is already verified on disk:
        if any(os.path.exists(os.path.join(output_dir, f"{vid}{ext}")) for ext in VALID_EXTS):
            verified_count += 1
            print(f"[Class {action_id} | {verified_count}/{target}] Verified on disk -> {vid}")
            continue

        url = f"https://www.youtube.com/watch?v={vid}"
        print(f"[Class {action_id} | {verified_count+1}/{target}] Fetching -> {vid} ...")

        try:

            # Instantiate a fresh session per video to prevent error state bleeding:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Verify physical file existence before incrementing success counters:
            if any(os.path.exists(os.path.join(output_dir, f"{vid}{ext}")) for ext in VALID_EXTS):
                verified_count += 1
                print(f"[Class {action_id} | {verified_count}/{target}] SUCCESS -> {vid}")
            else:
                print(f"[Class {action_id}] Format completely unavailable -> Blacklisting {vid}...")
                treated_ids.add(vid)
                add_to_blacklist(vid)

        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e).lower()

            # If YouTube is rate-limiting or throttling:
            transient_keywords = [
                "rate-limited",
                "try again later",
                "too many requests",
                "429",
                "bot",
                "captcha",
                "challenge",
            ]
            if any(tk in error_msg for tk in transient_keywords):
                print(f"[Class {action_id}] Rate-limit / Throttling hit -> Skipping WITHOUT blacklisting: {vid}")
                continue

            # Use strict full-length phrases to avoid rate-limit collisions:
            fatal_keywords = [
                "private",
                "terminated",
                "removed",
                "copyright",
                "violating",
                "requested format is not available",
                "no video formats found",
                "video unavailable",
                "we're processing this video"
            ]

            # If error is truly permanent link rot, add to blacklist:
            if any(fk in error_msg for fk in fatal_keywords):
                print(f"[Class {action_id}] Fatal Link Rot detected -> Blacklisting {vid}...")
                treated_ids.add(vid)
                add_to_blacklist(vid)

            # Otherwise treat as an unrecognized transient network glitch:
            else:
                print(f"[Class {action_id}] Unrecognized transient error -> Skipping for now: {vid}")

        # Tiny randomized sleep to prevent multi-threaded request flooding against YouTube servers:
        time.sleep(random.uniform(0.7, 1.5))

    print(f"[Worker -> Class {action_id}] Finished! Total videos ready: {verified_count}/{target}")
    return verified_count


def download_dataset(csv_list, output_dir="input/ava_kinetics/videos", videos_per_class=300, blacklist_path="dead_links.json"):
    """
    Parses multiple AVA and Kinetics annotation CSVs, standardizes their schemas, combines them into a master video pool, and launches a multi-threaded thread pool to download categories in parallel.
    :param csv_list: list of filepaths to the dataset CSV files.
    :param output_dir: destination directory to save the downloaded video files.
    :param videos_per_class: maximum number of valid video downloads required per action ID.
    :param blacklist_path: path to the JSON ledger file tracking dead links.
    :return: total number of videos successfully downloaded or verified across all categories.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load the blacklist into memory:
    treated_ids, disk_ids = init_blacklist(output_dir, blacklist_path)
    with open(blacklist_path, 'r') as f:
        dead_ids = set(json.load(f))

    # Standardized 8-column header (accommodates both 7-col Kinetics and 8-col AVA formats):
    col_names = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "person_id"]
    df_list = []

    # Parse and filter spreadsheets:
    print(f"Loading and merging {len(csv_list)} annotation spreadsheets...")
    for csv_path in csv_list:
        if not os.path.exists(csv_path):
            print(f"Warning: File not found -> '{csv_path}'. Skipping...")
            continue

        print(f"  -> Parsing: {csv_path}")
        df_temp = pd.read_csv(csv_path, header=None, names=col_names, low_memory=False)
        df_temp = df_temp.dropna(subset=["action_id"])
        df_list.append(df_temp)

    # Verify at least one file loaded successfully:
    if not df_list:
        print("Error: No valid CSV files were loaded. Exiting.")
        return 0

    # Combine all spreadsheets into a single master pool:
    df = pd.concat(df_list, ignore_index=True)
    df["action_id"] = df["action_id"].astype(int)

    # Remove duplicate video IDs within the same action class:
    df = df.drop_duplicates(subset=["action_id", "video_id"])
    print(f"Master pool assembled! Total unique action-to-video mappings available: {len(df)}")

    # Set downloader options with multi-client fallback chain:
    ydl_opts = {
        'format': 'best[height<=480]/bestvideo[height<=480]/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'cookiesfromfile': 'cookies.txt',
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': False,
        'js_runtimes': {
            'node': {}
        },
        'extractor_args': {
            'youtube': {
                'player_client': ['ios', 'android', 'tv_embedded', 'web']
            }
        },
        'sleep_interval': 2,
        'max_sleep_interval': 7,
        'sleep_interval_requests': 1.5
    }

    print(f"\nLaunching parallel download pool across {MAX_WORKERS} concurrent category workers...\n")

    # Launch thread pool to process each target action class concurrently:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for action_id in TARGET_CLASSES:

            # Extract unique YouTube video IDs for this specific class:
            class_vids = df[df['action_id'] == action_id]['video_id'].unique().tolist()
            
            # Calculate how many physical files already exist on disk for this class:
            verified_count = sum(1 for vid in class_vids if vid in disk_ids)
            
            # Submit worker with the unified treated set and initial verified count:
            futures.append(executor.submit(
                _download_class_worker, 
                action_id, class_vids, treated_ids, verified_count, videos_per_class, output_dir, ydl_opts
            ))

        # Gather completed video counts from all thread workers:
        results = [future.result() for future in futures]

    # Calculate total downloaded files:
    total_downloaded = sum(results)
    print(f"\n=======================================================")
    print(f"Parallel Batch Download Complete!")
    print(f"Total video files across all classes: {total_downloaded}")
    print(f"=======================================================")
    return total_downloaded


if __name__ == "__main__":
    try:
        download_dataset(
            csv_list=["kinetics_train_v1.0.csv", "kinetics_val_v1.0.csv", "ava_train_v2.2.csv", "ava_val_v2.2.csv"],
            output_dir="input/ava_kinetics/videos",
            videos_per_class=300
        )
    except Exception as e:
        print(e)
        os._exit(1)