import pandas as pd
import requests
import re

def download_file(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully to {destination}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def hu_data_loader(address='http://humap2.proteincomplexes.org/static/downloads/humap2/humap2_complexes_20200809.txt'):
  file_url = 'http://humap2.proteincomplexes.org/static/downloads/humap2/humap2_complexes_20200809.txt'
  destination_path = file_url.split('/')[-1]
  download_file(file_url,destination_path)


  data=pd.read_csv(destination_path)
  return data

def query_hu_data(data,name):
    returned_list=list(data[data['genenames'].apply(lambda x: name in x.split( ))]['genenames'])
    returned_list=' '.join(returned_list)
    returned_list=returned_list.replace(name,'')
    returned_list=re.sub(r'\s+', ' ', returned_list)
    returned_list=returned_list.split(' ')
    return returned_list

