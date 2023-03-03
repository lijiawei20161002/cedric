import json
import csv

links = []
country_links = set()

with open("cycle-aslinks.l8.20221001.1664586221.gva-ch.txt") as f:
    content = f.readlines()
    for l in content:
        if l.split()[0] == 'D':
            links.append([l.split()[1].split('_')[0].split(',')[0], l.split()[2].split('_')[0].split(',')[0]])
    f.close()

with open("links.txt","a+") as f:
    for link in links:
        f.writelines(str(link[0])+'\t'+str(link[1])+'\n')
    f.close()

with open("dns.txt") as f:
    dns2country = {}
    content = f.readlines()
    for l in content:
        dns2country[l.split()[0]] = l.split(",")[1].split()[0]

for link in links:
    if link[0] in dns2country and link[1] in dns2country:
        country_links.add((dns2country[link[0]], dns2country[link[1]]))

with open("link.txt", "a+") as f:
    for c in country_links:
        f.write(c[0]+','+c[1]+'\n')

with open("ddos.json", "r") as f:
    data = json.load(f)
    f.close()

keys = data['biggest']['attacks'][0].keys()
with open("attack.csv", "w") as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data['biggest']['attacks'])
