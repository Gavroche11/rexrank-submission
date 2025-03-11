import json
from typing import Tuple, List, Dict

def extract_age_and_gender(context: str) -> Tuple[str, str]:
    if not context.startswith("Age:"):
        age_value, gender_value = "N/A", "N/A"
    else:
        age, gender = context.split(".")[:2]
        # Check age
        try:
            age = age.replace("Age:", "").strip()

            # Check if age is in the expected format (e.g. "40-50")
            if "-" in age:
                age_inf, age_sup = age.split("-")
                age_value = str((int(age_inf) + int(age_sup)) // 2)
            else:
                age_value = "N/A"
        except Exception:
            age_value = "N/A"

        # Check gender
        try:
            gender = gender.replace("Gender:", "").strip()
            if gender == "F":
                gender_value = "Female"
            elif gender == "M":
                gender_value = "Male"
            else:
                gender_value = "N/A"
        except Exception:
            gender_value = "N/A"

    return age_value, gender_value

def make_right_format(input_json_file: str) -> List[Dict[str, str]]:
    raw_dataset = json.load(open(input_json_file))

    new_dataset = []

    for study_id, data in raw_dataset.items():
        # image_path = data["image_path"]
        # view_positions = data["frontal_lateral"]
        key_image_path = data["key_image_path"]
        context = data["context"]
        gt = data["report"]

        # key_image_idx = image_path.index(key_image_path)
        # key_view_position = view_positions[key_image_idx]

        age, gender = extract_age_and_gender(context)

        new_dataset.append({
            "study_id": study_id,
            "image_path": key_image_path,
            "age": age,
            "gender": gender,
            "gt": gt
        })

    return new_dataset