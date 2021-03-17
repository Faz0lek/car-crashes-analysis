"""IZV project part 1

This module implements DataDownloader class.
"""

__author__ = "Martin Kostelník (xkoste12)"

import os
import sys
import requests
import csv
import pickle
import gzip
from io import TextIOWrapper
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from zipfile import ZipFile
import numpy as np

class DataDownloader:
    """This class implements downloading car accidents data and its parsing."""
    
    def __init__(self, url="https://ehw.fit.vutbr.cz/izv/", folder="data", cache_filename="data_{}.pkl.gz"):
        """Initialize DataDownloader instance

        Keyword arguments:
        url -- Data will be downloaded from this URL. (default https://ehw.fit.vutbr.cz/izv/)
        folder -- Data will be saved in this folder. Use absolute paths or multiple folders at your own risk (default data)
        cache_filename -- Name of caching files. Use without {} or with mupliple brackets at your own risk (default data_{}.pkl.gz)
        """

        self.folder = folder
        self.url = url
        self.cache_filename = cache_filename

        self.data_archives = ["datagis2016.zip", "datagis-rok-2017.zip", "datagis-rok-2018.zip", "datagis-rok-2019.zip", "datagis-09-2020.zip"]
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"}
        self.region_cache = dict()

        self.col_headers = ["p1", "p36", "p37", "p2a", "weekday(p2a)", "p2b", "p6", "p7", "p8", "p9", "p10",
                            "p11", "p12", "p13a", "p13b", "p13c", "p14", "p15", "p16", "p17", "p18", "p19",
                            "p20", "p21", "p22", "p23", "p24", "p27", "p28", "p34", "p35", "p39", "p44",
                            "p45a", "p47", "p48a", "p49", "p50a", "p50b", "p51", "p52", "p53", "p55a",
                            "p57", "p58", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n",
                            "o", "p", "q", "r", "s", "t", "p5a", "region"]

        self.data_types = ["int", "int", "int", "U64", "int", "int", "int", "int", "int", "int",
                           "int", "int", "int", "int", "int", "int", "int", "int", "int", "int", 
                           "int", "int", "int", "int", "int", "int", "int", "int", "int", "int", 
                           "int", "int", "int", "int", "U64", "int", "int", "int", "int", "int", 
                           "int", "int", "int", "int", "int", "float", "float", "float", "float", "float", 
                           "float", "U64", "U64", "U64", "U64", "U64", "U64", "float", "U64", "U64", 
                           "U64", "int", "U64", "int"] # region data type is missing as it is added later

        self.region_files = { "PHA": "00.csv",    # Praha
                              "STC": "01.csv",    # Středočeský kraj
                              "JHC": "02.csv",    # Jihočeský kraj
                              "PLK": "03.csv",    # Plzeňský kraj
                              "ULK": "04.csv",    # Ústecký kkraj
                              "HKK": "05.csv",    # Královéhradecký kraj
                              "JHM": "06.csv",    # Jihomoravský kraj
                              "MSK": "07.csv",    # Moravskoslezský kraj
                              "OLK": "14.csv",    # Olomoucký kraj
                              "ZLK": "15.csv",    # Zlínský kraj
                              "VYS": "16.csv",    # Vysočina
                              "PAK": "17.csv",    # Pardubický kraj
                              "LBK": "18.csv",    # Liberecký kraj
                              "KVK": "19.csv", }  # Karlovarský kraj

    def check_folder(self):
        """Checks if data folder exists, if not, creates it"""

        if not os.path.isdir(self.folder):
            try:
                os.mkdir(self.folder)
            except OSError:
                print("ERROR: could not create directory, quitting", file=sys.stderr)
                sys.exit(1)

    def download_data(self):
        """This method downloads data archives"""

        self.check_folder()

        # Create requests session
        with requests.Session() as s:
            s.headers.update(self.headers)
            response = s.get(self.url) # Get HTML

            links = BeautifulSoup(response.text, "html.parser").find_all('a') # Find all links
        
            for link in links[3::2]: # We only iterate through zip files (they start on index 3 and then apper once in 2 iterations)
                if link["href"][5:] in self.data_archives: # Compare link name with the archive we want (we only want relevant archives)
                    print(f"DOWNLOADING FILE: \"{link['href'][5:]}\"", file=sys.stderr)
                    zip_file = s.get(urljoin(self.url, link["href"]))
                    open(link["href"], "wb").write(zip_file.content)

    def parse_region_data(self, region):
        """Parse data for a specific region. This method also downloads data archives which are missing.


        Arguments:
        region -- region acronym

        Returns:
        Returns a tuple of two elements. First being column headers (list), the second being a list of ndarrays containing data.
        """

        self.check_folder()

        # Check if data files are present
        for archive in self.data_archives:
            if not os.path.isfile(f"./{self.folder}/{archive}"): # Data file not present, download it
                print(f"Data archive missing: {archive}. Downloading now.", file=sys.stderr)

                r = requests.get(self.url, headers=self.headers) # Get HTML
                links = BeautifulSoup(r.text, "html.parser").find_all('a') # Find all links

                for link in links[3::2]:
                    if archive == link["href"][5:]:
                        zip_file = requests.get(urljoin(self.url, link["href"]))
                        open(link["href"], "wb").write(zip_file.content)

        data_list = list()

        # Read archives
        for archive in self.data_archives:
            with ZipFile(f"./{self.folder}/{archive}", 'r') as zip:
                with zip.open(self.region_files[region], 'r') as data_file:
                    csv_data = csv.reader(TextIOWrapper(data_file, "cp1250"), delimiter=';', quotechar='"')
                    data_list.extend(list(csv_data))

        np_data = list()
        l = len(data_list)
        col_index = 0

        # Create np arrays using correct data type and fill them with data
        for t in self.data_types:
            np_data.append(np.ndarray(shape = (l), dtype=t)) # create np array to store data in

            for i in range(l):
                if t == "U64": # String
                    np_data[col_index][i] = data_list[i][col_index]
                else: # Int or Float
                    try:
                        if t == "int":
                            np_data[col_index][i] = data_list[i][col_index]
                        elif t == "float":
                            np_data[col_index][i] = data_list[i][col_index].replace(',', '.')
                    except ValueError:
                        np_data[col_index][i] = -9999 # -9999 is used as ERROR value
                    
            col_index += 1

        # Create the last np array containing region acronym
        np_data.append(np.full(shape=(l), dtype="U64", fill_value=region))

        return (self.col_headers, np_data)

    def get_list(self, regions=None):
        """This method aggregates data of several regions.

        Keyword arguments:
        regions -- list of region acronyms

        Returns:
        Returns a tuple of two elements. First being column headers (list), 
        the second being a list of ndarrays containing aggregated data. 
        """

        np_data = list()

        if regions is None:
            regions = self.region_files

        # Start caching or process data
        for region in regions:
            cache_file_path = f"./{self.folder}/{self.cache_filename.format(region)}"

            if region in self.region_cache.keys(): # Result is in memory
                pass
            elif os.path.isfile(cache_file_path): # Result is NOT in memory, but IS in cache file
                with gzip.open(cache_file_path, "rb") as gfile:
                    print(f"Loading {region} region data from cache file: {cache_file_path[7:]}", file=sys.stderr)
                    self.region_cache[region] = pickle.load(gfile)
            else: # Result is NEITHER in memory NOR cache file
                # add to memory
                self.region_cache[region] = self.parse_region_data(region)[1]
                # add to cache
                print(f"Adding {region} region data to cache file: {cache_file_path[7:]}", file=sys.stderr)
                with gzip.open(cache_file_path, "wb") as gfile:
                    pickle.dump(self.region_cache[region], gfile)

        # Concatenate region data
        for i in range(65):
            np_data.append(np.concatenate([x[i] for _, x in self.region_cache.items()]))

        return (self.col_headers, np_data)

if __name__ == "__main__":
    data = DataDownloader().get_list(["MSK", "JHM", "ZLK"])
    print("Kraje: MSK, JHM, ZLK")
    print(f"Počet záznamů: {len(data[1][0])}")
    print(f"Sloupce: {data[0]}")
