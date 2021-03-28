import bs4
import requests
from requests import status_codes
import pandas as pd
import textwrap
from bs4 import BeautifulSoup as soup

papers = pd.read_csv('dataset.csv')
url = pd.DataFrame(papers, columns=['GTRProjectUrl', 'FundingOrgName'])
epsrc = url.loc[url['FundingOrgName'] == "EPSRC"]
abstracts = epsrc['GTRProjectUrl'].values

filename = "abstract1.csv"
f = open(filename, "w")
header = "All Abstracts"
f.write(header)

for abstract in abstracts:
    result = requests.get(abstract)
    if result.status_code == requests.codes.ok:
        src = result.content
        extract = soup(src, 'html.parser')
        content = extract.find("gtr:abstracttext")
        title = extract.find("gtr:title")
        funding = extract.find("gtr:valuepounds")
        print("Project Title: " + title.string)
        print("Funding Organization: EPSRC")
        print("Project Abstract: " + textwrap.fill(content.string), end='\n')
        print("Funded Value : £" + funding.string)
        print(len(abstracts))
        print(abstract)

        f.write("\n" + content.string.replace(",", "|"))

f.close()







