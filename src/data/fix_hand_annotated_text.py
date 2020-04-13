import glob
import re
import sys

from tqdm import tqdm

"The docs annotated by MM where missing the closing tag. We fix that here"
if len(sys.argv) < 3:
    print("You need to pass a folder with the text files to fix and whether these files are annotated or not")
    exit(1)

path = sys.argv[1]
is_annotated = sys.argv[2]
if is_annotated == "annotated":
    is_annotated = True
else:
    is_annotated = False

list_files = glob.glob(path + "/**/*.txt", recursive=True)
# list_files = ["/data/conseil_etat/manual_martinie/hand_annotated/Annotees_lot2/421443.txt"]
for f in tqdm(list_files):
    with open(f, "r", encoding="utf-8-sig") as txt:
        content = txt.read()
    if is_annotated:
        content = re.sub(r"<(/?)adr>", "<\g<1>ADRESSE>", content)
        content = re.sub(r"<NOM><nom>LE CORFF<NOM></nom><ano>X</ano>", "<NOM>LE CORFF</NOM>", content)
        content = re.sub(r"<NOM><NOM>BOURI D'HAROUB</NOM> d'HAROUB<NOM>", "<NOM>BOURI D'HAROUB</NOM>", content, re.IGNORECASE)
        content = re.sub(r"(\<NOM\>[\w\s\-'’]+)\<NOM\>", r'\g<1></NOM>', content)
        content = re.sub(r"(\<PRENOM\>[\w\s\-'’]+)\<PRENOM\>", r'\g<1></PRENOM>', content)
        content = re.sub(r"(\<ADRESSE\>[\w\s\-'’]+)\<ADRESSE\>", r'\g<1></ADRESSE>', content)

    else:
        # 421443
        content = re.sub(r"Bouri", "Bouri d'Haroub", content)
        content = re.sub(r"<nom>LE CORFF</nom><ano>X</ano>", "LE CORFF", content)

    content = re.sub(r"</?adr>", "", content)
    content = re.sub(r"’", "'", content)
    content = re.sub(r"celle du décès de M\. Abriata la", "celle du décès de M. Abriata, la", content)

    with open(f, "w") as txt:
        txt.write(content)
