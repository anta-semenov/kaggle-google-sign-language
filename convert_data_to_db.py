import os
import pandas as pd
import sqlite3

# Define the path to the top-level folder
top_level_folder = './asl-signs/train_landmark_files'

maxFramesCount = 0
minFramesCount = 100000000000000
# Loop through the top-level folders
for participant_folder in os.listdir(top_level_folder):
    if not participant_folder.endswith('.DS_Store'):
      participant_id = int(participant_folder)
      print('Record participant ID:', participant_id)
      participant_path = os.path.join(top_level_folder, participant_folder)

      # Loop through the second-level folders
      for sequence_file in os.listdir(participant_path):
        if sequence_file.endswith('.parquet'):
          sequence_id = int(os.path.splitext(sequence_file)[0])
          sequence_path = os.path.join(participant_path, sequence_file)
          df = pd.read_parquet(sequence_path)
          # print('Frames count:', df['frame'].unique().size, 'for sequence ID:', sequence_id, 'and participant ID:', participant_id)
          maxFramesCount = max(maxFramesCount, df['frame'].unique().size)
          minFramesCount = min(minFramesCount, df['frame'].unique().size)

print('Max frames count:', maxFramesCount)
print('Min frames count:', minFramesCount)

          