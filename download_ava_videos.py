import os
import pandas as pd
import yt_dlp

# The 10 specific AVA classes selected:
TARGET_CLASSES = [5, 8, 10, 11, 12, 14, 27, 29, 54, 57]


def download_dataset(csv_list, output_dir="input/ava_kinetics/videos", videos_per_class=300):
    """
    Parses multiple AVA and Kinetics annotation CSVs, standardizes their column schemas, combines them into a unified video pool, and downloads the raw YouTube footage.
    :param csv_list: list of filepaths to the dataset CSV files.
    :param output_dir: destination directory to save the downloaded video files.
    :param videos_per_class: maximum number of valid video downloads required per action ID.
    :return: total number of videos successfully downloaded or verified on disk.
    """

    # Create output directory:
    os.makedirs(output_dir, exist_ok=True)

    # Standardized 8-column header:
    col_names = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "person_id"]
    df_list = []

    # Scan spreadsheets:
    print(f"Loading and merging {len(csv_list)} annotation spreadsheets...")
    for csv_path in csv_list:
        if not os.path.exists(csv_path):
            print(f"Warning: File not found -> '{csv_path}'. Skipping...")
            continue
        # Read ragged CSVs safely:
        print(f"  -> Parsing: {csv_path}")
        df_temp = pd.read_csv(csv_path, header=None, names=col_names, low_memory=False)

        # Drop 2-column empty frame rows where action_id is NaN:
        df_temp = df_temp.dropna(subset=["action_id"])
        df_list.append(df_temp)
    if not df_list:
        print("Error: No valid CSV files were loaded. Exiting.")
        return 0

    # Combine all spreadsheets into a single master pool:
    df = pd.concat(df_list, ignore_index=True)
    df["action_id"] = df["action_id"].astype(int)

    # Remove duplicate video IDs within the same action class to prevent downloading the same link twice:
    df = df.drop_duplicates(subset=["action_id", "video_id"])

    # Configure yt-dlp to download lightweight MP4/WebM files and ignore dead links:
    ydl_opts = {
        'format': 'bestvideo[height<=480]',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True
    }

    # For each action:
    for action_id in TARGET_CLASSES:
        print(f"\n--- Downloading videos for AVA Action ID: {action_id} ---")

        # Get all unique YouTube video IDs associated with this action:
        class_vids = df[df['action_id'] == action_id]['video_id'].unique()
        print(f"Found {len(class_vids)} videos in {action_id}.")
        downloaded = 0
        for vid in class_vids:

            # Stop once we have enough videos for this class to train the network:
            if downloaded >= videos_per_class:
                break

            # Check if video was already downloaded in a previous run:
            vid_path_mp4 = os.path.join(output_dir, f"{vid}.mp4")
            vid_path_webm = os.path.join(output_dir, f"{vid}.webm")
            if os.path.exists(vid_path_mp4) or os.path.exists(vid_path_webm):
                print(f"[{downloaded + 1}/{videos_per_class}] Already exists: {vid}")
                downloaded += 1
                continue

            # Try downloading:
            url = f"https://www.youtube.com/watch?v={vid}"
            print(f"[{downloaded + 1}/{videos_per_class}] Fetching {url} ...")
            try:

                # Instantiate a new session every time to prevent error state bleeding:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                # Only increment if the file physically exists on the hard drive:
                if os.path.exists(vid_path_mp4) or os.path.exists(vid_path_webm) or os.path.exists(vid_path_mkv):
                    downloaded += 1
                else:
                    print(f"  -> Video deleted or unavailable on YouTube. Skipping...")

            except Exception as e:
                print(f"  -> Error encountered. Skipping...")

    print(f"\nDownload complete! Videos are saved in '{output_dir}'.")


if __name__ == "__main__":
    download_dataset(
        csv_list=["kinetics_train_v1.0.csv", "kinetics_val_v1.0.csv", "ava_train_v2.2.csv", "ava_val_v2.2.csv"],
        output_dir="input/ava_kinetics/videos",
        videos_per_class=300
    )
