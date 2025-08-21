import os

# Path where your numbered folders are
base_path = r"D:\my projects\scrip_translator\Data\set"

# Mapping numbers (1–49) to Kannada letters
num_to_kannada = {
    1: 'ಅ', 2: 'ಆ', 3: 'ಇ', 4: 'ಈ', 5: 'ಉ', 6: 'ಊ', 7: 'ಋ', 8: 'ೠ',
    9: 'ಌ', 10: 'ೡ', 11: 'ಎ', 12: 'ಏ', 13: 'ಐ', 14: 'ಒ', 15: 'ಓ', 16: 'ಔ',
    17: 'ಅಂ', 18: 'ಅಃ', 19: 'ಕ', 20: 'ಖ', 21: 'ಗ', 22: 'ಘ', 23: 'ಙ',
    24: 'ಚ', 25: 'ಛ', 26: 'ಜ', 27: 'ಝ', 28: 'ಞ',
    29: 'ಟ', 30: 'ಠ', 31: 'ಡ', 32: 'ಢ', 33: 'ಣ',
    34: 'ತ', 35: 'ಥ', 36: 'ದ', 37: 'ಧ', 38: 'ನ',
    39: 'ಪ', 40: 'ಫ', 41: 'ಬ', 42: 'ಭ', 43: 'ಮ',
    44: 'ಯ', 45: 'ರ', 46: 'ಲ', 47: 'ವ',
    48: 'ಶ', 49: 'ಷ'
}

# Rename folders
for num, kannada_char in num_to_kannada.items():
    old_folder = os.path.join(base_path, str(num))
    new_folder = os.path.join(base_path, kannada_char)
    if os.path.exists(old_folder):
        os.rename(old_folder, new_folder)
        print(f"Renamed {old_folder} -> {new_folder}")
    else:
        print(f"Folder {old_folder} not found, skipping...")

