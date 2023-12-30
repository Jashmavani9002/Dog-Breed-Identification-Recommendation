import requests
from bs4 import BeautifulSoup
import re

# List of dog breeds
dog_breeds = [
    "affenpinscher", "afghan_hound", "african_hunting_dog", "airedale", "american_staffordshire_terrier",
    "appenzeller", "australian_terrier", "basenji", "basset", "beagle", "bedlington_terrier",
    "bernese_mountain_dog", "black-and-tan_coonhound", "blenheim_spaniel", "bloodhound", "bluetick",
    "border_collie", "border_terrier", "borzoi", "boston_bull", "bouvier_des_flandres", "boxer",
    "brabancon_griffon", "briard", "brittany_spaniel", "bull_mastiff", "cairn", "cardigan", "chesapeake_bay_retriever",
    "chihuahua", "chow", "clumber", "cocker_spaniel", "collie", "curly-coated_retriever", "dandie_dinmont", "dhole",
    "dingo", "doberman", "english_foxhound", "english_setter", "english_springer", "entlebucher", "eskimo_dog",
    "flat-coated_retriever", "french_bulldog", "german_shepherd", "german_short-haired_pointer", "giant_schnauzer",
    "golden_retriever", "gordon_setter", "great_dane", "great_pyrenees", "greater_swiss_mountain_dog", "groenendael",
    "ibizan_hound", "irish_setter", "irish_terrier", "irish_water_spaniel", "irish_wolfhound", "italian_greyhound",
    "japanese_spaniel", "keeshond", "kelpie", "kerry_blue_terrier", "komondor", "kuvasz", "labrador_retriever",
    "lakeland_terrier", "leonberg", "lhasa", "malamute", "malinois", "maltese_dog", "mexican_hairless",
    "miniature_pinscher", "miniature_poodle", "miniature_schnauzer", "newfoundland", "norfolk_terrier",
    "norwegian_elkhound", "norwich_terrier", "old_english_sheepdog", "otterhound", "papillon", "pekinese", "pembroke",
    "pomeranian", "pug", "redbone", "rhodesian_ridgeback", "rottweiler", "saint_bernard", "saluki", "samoyed",
    "schipperke", "scotch_terrier", "scottish_deerhound", "sealyham_terrier", "shetland_sheepdog", "shih-tzu",
    "siberian_husky", "silky_terrier", "soft-coated_wheaten_terrier", "staffordshire_bullterrier", "standard_poodle",
    "standard_schnauzer", "sussex_spaniel", "tibetan_mastiff", "tibetan_terrier", "toy_poodle", "toy_terrier",
    "vizsla", "walker_hound", "weimaraner", "welsh_springer_spaniel", "west_highland_white_terrier", "whippet",
    "wire-haired_fox_terrier", "yorkshire_terrier"
]

# Function to get the image URL for a given dog breed
def get_dog_breed_image_url(breed):
    search_url = f"https://www.google.com/search?q={breed}+dog&tbm=isch"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

    # Send a GET request to Google Images
    response = requests.get(search_url, headers=headers)

    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the image URL from the search results
    img_tags = soup.find_all("img", class_="t0fcAb")
    if img_tags:
        img_url = img_tags[0]["src"]
        return img_url

    return None

# Fetch and print image URLs for each dog breed
for breed in dog_breeds:
    img_url = get_dog_breed_image_url(breed)
    print(f"{breed}: {img_url}")
