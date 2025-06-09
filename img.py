import json
import os

def update_inventory_settings():
    """
    Updates the Inventory Settings JSON file with sequential image links.
    Pre-configured with your specific values.
    """
    # Your specific configuration
    file_path = r"C:\Users\hp\Downloads\final\data\Requests Settings.json"
    base_link = "https://github.com/SavvyGaikwad/img/tree/main/Requests%20Settings/"
    max_number = 17
    exclude_numbers = []
    
    print("Inventory Settings JSON Updater")
    print("=" * 50)
    print(f"File: {file_path}")
    print(f"Base link: {base_link}")
    print(f"Max number: {max_number}")
    print(f"Excluding: {exclude_numbers}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        return False
    
    # Read the JSON file
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print("✅ JSON file loaded successfully")
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    # Generate sequence of numbers excluding specified ones
    available_numbers = [i for i in range(1, max_number + 1) if i not in exclude_numbers]
    print(f"Available numbers: {available_numbers}")
    
    # Counter for tracking current image number
    current_number_index = 0
    
    def update_media_recursive(obj):
        nonlocal current_number_index
        
        if isinstance(obj, dict):
            # If this is a media object, update its path
            if obj.get("type") == "media" and "path" in obj:
                if current_number_index < len(available_numbers):
                    number = available_numbers[current_number_index]
                    new_path = f"{base_link}{number}.png"
                    old_path = obj["path"]
                    obj["path"] = new_path
                    current_number_index += 1
                    print(f"  Updated: {old_path} → {new_path}")
            
            # Recursively process all values in the dictionary
            for key, value in obj.items():
                update_media_recursive(value)
        
        elif isinstance(obj, list):
            # Recursively process all items in the list
            for item in obj:
                update_media_recursive(item)
    
    # Create backup of original file
    backup_path = file_path.replace('.json', '_backup.json')
    try:
        with open(file_path, 'r', encoding='utf-8') as original:
            with open(backup_path, 'w', encoding='utf-8') as backup:
                backup.write(original.read())
        print(f"✅ Backup created: {backup_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not create backup: {e}")
    
    # Update the media links
    print("\nUpdating media links:")
    print("-" * 30)
    update_media_recursive(data)
    
    # Write the updated JSON back to file
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        print("-" * 30)
        print(f"✅ Successfully updated '{file_path}'")
        print(f"📊 Total media sections updated: {current_number_index}")
        return True
    except Exception as e:
        print(f"❌ Error writing to file: {e}")
        return False

def main():
    """
    Main function with confirmation prompt.
    """
    print("This will update your Inventory Settings.json file with sequential image links.")
    print("A backup will be created automatically.")
    print()
    
    proceed = input("Do you want to proceed? (y/n): ").strip().lower()
    
    if proceed in ['y', 'yes']:
        print()
        success = update_inventory_settings()
        if success:
            print("\n🎉 Update completed successfully!")
            print("Your original file has been backed up.")
        else:
            print("\n💥 Update failed! Check the error messages above.")
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()