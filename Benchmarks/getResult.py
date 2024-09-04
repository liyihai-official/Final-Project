#imports 
import os
import sys
import pandas as pd


def getOutputs(path_to_outputs : str) -> [tuple, tuple, tuple]:
  script, jobid=[], []
  files=[
    f for f in os.listdir(path_to_outputs) if os.path.isfile(
      os.path.join(path_to_outputs, f)
    )
  ]
  
  for f in files:
    other = f.split('.')
    jobid.append(int(other[-1]))
    script.append(other[0])
    
  return files, list(set(script)), sorted(jobid)


def getResult(path_to_result : str):
  data=[] 
  with open(path_to_result, 'r') as file:
    entry = {}
    for line in file:
      line = line.strip()

      if line.startswith("Rows:"):
        entry['Rows'] = int(line.split(":")[1].strip()) + 2
      elif line.startswith("Columns:"):
        entry['Columns'] = int(line.split(":")[1].strip()) + 2
      elif line.startswith("Depths:"):
        entry['Depths'] = int(line.split(":")[1].strip()) + 2
      elif line.startswith("Number of MPI Processes:"):
        entry['MPI Processes'] = int(line.split(":")[1].strip())
      elif line.startswith("OpenMP Threads:"):
        entry['OpenMP Threads'] = int(line.split(":")[1].strip())
      elif line.startswith("Total Converge time:"):
        entry['Total Converge time'] = float(line.split(":")[1].strip())
      elif line.startswith("Iterations:"):
        entry['Iterations'] = int(line.split(":")[1].strip())
      elif line.startswith("===="):
        if entry:
          data.append(entry)
          entry = {}
    
    # Add last entry if the file doesn't end with '===='
    if entry:
      data.append(entry)

  return data

def Preprocessing(df : pd.DataFrame):
    Dimension=2
    Depths=[]
    Rows = df["Rows"].unique().astype(int)
    Cols = df["Columns"].unique().astype(int)
    if "Depths" in df.columns:
        Depths=df["Depths"].unique().astype(int)
        Dimension=3
    
    df["Processes"] = df["MPI Processes"] * df["OpenMP Threads"]
    
    Processes=df["Processes"].unique().astype(int)
    Threads=df["OpenMP Threads"].unique().astype(int)
    
    return Rows, Cols, Dimension, Processes, Threads, Depths, df

def main():
  all_files, all_scripts, all_jobids=getOutputs(path_to_outputs=path_to_outputs)
  print(all_scripts, all_jobids)
  print(all_files)

  data = pd.DataFrame(getResult('output/single.o.4474865'))
  data = data.dropna()
  Rows, Cols, Dimension, Processes, Threads, Depths, data = Preprocessing(data)







if __name__ == "__main__":
  path_to_outputs='output/'
  all_files, all_jobids, all_scripts=[], [], []
  main()