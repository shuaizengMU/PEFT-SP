# © Copyright Technical University of Denmark
"""
Collection of PHYLUM identifiers from Uniprot.
Apply get_kingdom() to a list of PHYLUM IDs to get
the organism group that is used by the model.
"""

positive = [
    "Firmicutes",
    "Actinobacteria",
    "Thermotogae",
    "Chloroflexi",
    "Saccharibacteria",
    "Tenericutes",
]
negative = [
    "Candidatus Taylorbacteria",
    "Gemmatimonadetes",
    "Candidatus Kryptonia",
    "Candidatus Marinimicrobia",
    "Chloroflexi",
    "Candidatus Fraserbacteria",
    "Rhodothermaeota",
    "Candidatus Coatesbacteria",
    "Candidatus Giovannonibacteria",
    "Candidatus Abawacabacteria",
    "Candidatus Sumerlaeota",
    "Candidatus Peregrinibacteria",
    "Candidatus Ryanbacteria",
    "Candidatus Riflebacteria",
    "Candidatus Cryosericota",
    "Proteobacteria",
    "Balneolaeota",
    "Deinococcus-Thermus",
    "Nitrospinae",
    "Candidatus Colwellbacteria",
    "Chlorobi",
    "Candidatus Goldbacteria",
    "Kiritimatiellaeota",
    "Candidatus Microgenomates",
    "Candidatus Handelsmanbacteria",
    "Candidatus Aureabacteria",
    "Candidatus Cerribacteria",
    "Candidatus Gracilibacteria",
    "Candidatus Liptonbacteria",
    "Fusobacteria",
    "Candidatus Saccharibacteria",
    "Candidatus Magasanikbacteria",
    "Candidatus Shapirobacteria",
    "Candidatus Doudnabacteria",
    "Candidatus Andersenbacteria",
    "Calditrichaeota",
    "Candidatus Curtissbacteria",
    "Candidatus Yanofskybacteria",
    "Cyanobacteria",
    "Candidatus Wirthbacteria",
    "Firmicutes",
    "Candidatus Eisenbacteria",
    "candidate division CPR1",
    "Armatimonadetes",
    "Candidatus Firestonebacteria",
    "Candidatus Terrybacteria",
    "Candidatus Zambryskibacteria",
    "Candidatus Kuenenbacteria",
    "Candidatus Campbellbacteria",
    "Candidatus Beckwithbacteria",
    "Candidatus Raymondbacteria",
    "Candidatus Nealsonbacteria",
    "Candidatus Brennerbacteria",
    "Caldiserica",
    "Candidatus Vogelbacteria",
    "Candidatus Margulisbacteria",
    "Candidatus Dormibacteraeota",
    "Candidatus Uhrbacteria",
    "Candidatus Poribacteria",
    "Candidatus Kapabacteria",
    "Candidatus Kaiserbacteria",
    "Candidatus Komeilibacteria",
    "Candidatus Desantisbacteria",
    "Candidatus Adlerbacteria",
    "Candidatus Parcubacteria",
    "Candidatus Nomurabacteria",
    "Candidatus Spechtbacteria",
    "Candidatus Fervidibacteria",
    "candidate division CPR3",
    "Candidatus Chisholmbacteria",
    "Actinobacteria",
    "Candidatus Kerfeldbacteria",
    "Candidatus Wallbacteria",
    "Coprothermobacterota",
    "candidate division Zixibacteria",
    "Candidatus Fermentibacteria",
    "Candidatus Veblenbacteria",
    "candidate division LCP-89",
    "Candidatus Omnitrophica",
    "Acidobacteria",
    "Candidatus Dadabacteria",
    "Candidatus Azambacteria",
    "Candidatus Cloacimonetes",
    "Candidatus Daviesbacteria",
    "Verrucomicrobia",
    "Chlamydiae",
    "Candidatus Harrisonbacteria",
    "candidate division KD3-62",
    "candidate division WOR-3",
    "candidate division CPR2",
    "Candidatus Woesebacteria",
    "Candidatus Jacksonbacteria",
    "Candidatus Tagabacteria",
    "Candidatus Aerophobetes",
    "Candidatus Lloydbacteria",
    "Candidatus Lindowbacteria",
    "Aquificae",
    "Spirochaetes",
    "candidate division JL-ETNP-Z39",
    "candidate division GAL15",
    "Candidatus Calescamantes",
    "Candidatus Sungbacteria",
    "candidate division WWE3",
    "Candidatus Collierbacteria",
    "candidate division FCPU426",
    "Candidatus Wolfebacteria",
    "Candidatus Portnoybacteria",
    "Candidatus Blackburnbacteria",
    "Candidatus Wildermuthbacteria",
    "Candidatus Fischerbacteria",
    "Candidatus Rokubacteria",
    "candidate division NC10",
    "Candidatus Schekmanbacteria",
    "Lentisphaerae",
    "Candidatus Aminicenantes",
    "Thermodesulfobacteria",
    "Thermotogae",
    "Candidatus Pyropristinus",
    "Fibrobacteres",
    "candidate division WPS-1",
    "Candidatus Staskawiczbacteria",
    "Candidatus Pacebacteria",
    "Candidatus Gottesmanbacteria",
    "Candidatus Abyssubacteria",
    "Candidatus Tectomicrobia",
    "Synergistetes",
    "Candidatus Melainabacteria",
    "Candidatus Falkowbacteria",
    "Candidatus Glassbacteria",
    "Nitrospirae",
    "Candidatus Roizmanbacteria",
    "Candidatus Moranbacteria",
    "Deferribacteres",
    "Bacteroidetes",
    "Candidatus Berkelbacteria",
    "Candidatus Hydrogenedentes",
    "Candidatus Eremiobacteraeota",
    "Candidatus Amesbacteria",
    "Candidatus Woykebacteria",
    "Candidatus Yonathbacteria",
    "Candidatus Atribacteria",
    "Ignavibacteriae",
    "Candidatus Edwardsbacteria",
    "Candidatus Delongbacteria",
    "Abditibacteriota",
    "Candidatus Bipolaricaulota",
    "Elusimicrobia",
    "Dictyoglomi",
    "Candidatus Hydrothermae",
    "Candidatus Mcinerneyibacteriota",
    "Candidatus Latescibacteria",
    "Candidatus Buchananbacteria",
    "Chrysiogenetes",
    "Tenericutes",
    "Candidatus Levybacteria",
    "Candidatus Niyogibacteria",
    "Planctomycetes",
    "Candidatus Jorgensenbacteria",
    "Krumholzibacteriota",
]
archaea = [
    "Crenarchaeota",
    "Nanoarchaeota",
    "Candidatus Nezhaarchaeota",
    "Candidatus Diapherotrites",
    "Candidatus Geothermarchaeota",
    "Candidatus Marsarchaeota",
    "Candidatus Lokiarchaeota",
    "Candidatus Woesearchaeota",
    "Candidatus Micrarchaeota",
    "Candidatus Aenigmarchaeota",
    "Candidatus Bathyarchaeota",
    "Candidatus Verstraetearchaeota",
    "Candidatus Thorarchaeota",
    "Candidatus Altiarchaeota",
    "Candidatus Korarchaeota",
    "Candidatus Helarchaeota",
    "Candidatus Hydrothermarchaeota",
    "Candidatus Heimdallarchaeota",
    "Candidatus Huberarchaea",
    "candidate phylum NAG2",
    "Candidatus Geoarchaeota",
    "Candidatus Parvarchaeota",
    "Thaumarchaeota",
    "Euryarchaeota",
    "Candidatus Odinarchaeota",
]
eukarya = [
    "Loricifera",
    "Rotifera",
    "Ctenophora",
    "Nematomorpha",
    "Priapulida",
    "Apicomplexa",
    "Acanthocephala",
    "Bacillariophyta",
    "Euglenozoa",
    "Olpidiomycota",
    "Picozoa",
    "Rhodophyta",
    "Platyhelminthes",
    "Parabasalia",
    "Haptista",
    "Heterolobosea",
    "Phoronida",
    "Chytridiomycota",
    "Evosea",
    "Arthropoda",
    "Nemertea",
    "Imbricatea",
    "Cryptomycota",
    "Zoopagomycota",
    "Mucoromycota",
    "Chlorophyta",
    "Fornicata",
    "Kinorhyncha",
    "Gnathostomulida",
    "Annelida",
    "Perkinsozoa",
    "Endomyxa",
    "Hemichordata",
    "Ascomycota",
    "Cercozoa",
    "Basidiomycota",
    "Microsporidia",
    "Porifera",
    "Hemimastigophora",
    "Ciliophora",
    "Cnidaria",
    "Preaxostyla",
    "Orthonectida",
    "Nematoda",
    "Tardigrada",
    "Placozoa",
    "Discosea",
    "Xenacoelomorpha",
    "Onychophora",
    "Tubulinea",
    "Echinodermata",
    "Dicyemida",
    "Chaetognatha",
    "Foraminifera",
    "Entoprocta",
    "Gastrotricha",
    "Streptophyta",
    "Brachiopoda",
    "Chordata",
    "Cycliophora",
    "Blastocladiomycota",
    "Bryozoa",
    "Mollusca",
    "Nematoda (roundworms)",
]


def get_kingdom(x: str):
    if x in positive:
        return "POSITIVE"
    elif x in negative:
        return "NEGATIVE"
    elif x in archaea:
        return "ARCHAEA"
    elif x in eukarya:
        return "EUKARYA"
    else:
        # print(f'drop {x}')
        return "UNKNOWN"