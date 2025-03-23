# Now process the dataset from .csv to text by extracting the title coloumn as a single per episode. 
# Then the next coloumn is the character that is speaking or a scene discription, 
# place these with [] brackets and then paste the line under this subheading. 
# After the line introduce a newline.

import pandas as pd
from alive_progress import alive_bar
import os

# navigate to content/nanoGPT/data/south_park_lines/ and list
os.chdir('/content/nanoGPT/data/southpark/')
print(os.getcwd())
print(os.listdir())

# Read the CSV file
df = pd.read_csv('SouthPark_Lines.csv')
# Print some statistics about the dataframe
print(df.describe())
# Print the first few rows of the dataframe
print(df.head())

# Open a TXT file to write in
with open('input.txt', 'w') as f:
  # Make a list with all the titles
  title_list = {}

  # Then check if the line is not a title in the list
  with alive_bar(len(df), force_tty=True) as bar:
    for i in range(len(df)):
      if df['Title'][i] not in title_list:
        # first write the episode title per epiode as a Heading
        f.write(f"# {df['Title'][0]}\n\n")

        # Append title to title list
        title_list[df['Title'][i]] = 1
      else:
        # then go over the lines in chronological order and first place the contents of the "Character" in square brackets
        f.write(f"[{df['Character'][i]}] ")
        # Then the line under it
        f.write(f"{df['Line'][i]}\n\n")
      
      #Print progress with a loading bar corresponding to the current line and the total length of df, permille
      if i % len(df)*0.001 == 0:
        bar()

# Close the document
f.close()