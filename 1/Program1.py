def display_menu():
    print("1. Operations on List")
    print("2. Operations on Tuple")
    print("3. Operations on Set")
    print("4. Operations on Dictionary")
    print("5. Exit")
    choice = int(input("Enter your choice: "))
    return choice

def list_operations():
    my_list = []
    while True:
        print("\nList Operations:")
        print("1. Insert")
        print("2. Update")
        print("3. Delete")
        print("4. Display")
        print("5. Sort")
        print("6. Search")
        print("7. Back to main menu")
        choice = int(input("Enter your choice: "))
        if choice == 1:
            item = input("Enter item to insert: ")
            my_list.append(item)
        elif choice == 2:
            index = int(input("Enter index to update: "))
            if index < len(my_list):
                new_item = input("Enter new item: ")
                my_list[index] = new_item
            else:
                print("Index out of range!")
        elif choice == 3:
            item = input("Enter item to delete: ")
            if item in my_list:
                my_list.remove(item)
            else:
                print("Item not found!")
        elif choice == 4:
            print("List:", my_list)
        elif choice == 5:
            my_list.sort()
            print("Sorted list:", my_list)
        elif choice == 6:
            item = input("Enter item to search: ")
            if item in my_list:
                print("Item found at index", my_list.index(item))
            else:
                print("Item not found!")
        elif choice == 7:
            break
        else:
            print("Invalid choice!")

def tuple_operations():
    my_tuple = ()
    while True:
        print("\nTuple Operations:")
        print("1. Display")
        print("2. Back to main menu")
        choice = int(input("Enter your choice: "))
        if choice == 1:
            print("Tuple:", my_tuple)
        elif choice == 2:
            break
        else:
            print("Invalid choice!")


def set_operations():
    my_set = set()
    while True:
        print("\nSet Operations:")
        print("1. Add")
        print("2. Remove")
        print("3. Display")
        print("4. Back to main menu")
        choice = int(input("Enter your choice: "))
        if choice == 1:
            item = input("Enter item to add: ")
            my_set.add(item)
        elif choice == 2:
            item = input("Enter item to remove: ")
            if item in my_set:
                my_set.remove(item)
            else:
                print("Item not found!")
        elif choice == 3:
            print("Set:", my_set)
        elif choice == 4:
            break
        else:
            print("Invalid choice!")


def dictionary_operations():
    my_dict = {}
    while True:
        print("\nDictionary Operations:")
        print("1. Insert")
        print("2. Update")
        print("3. Delete")
        print("4. Display")
        print("5. Search")
        print("6. Back to main menu")
        choice = int(input("Enter your choice: "))
        if choice == 1:
            key = input("Enter key: ")
            value = input("Enter value: ")
            my_dict[key] = value
        elif choice == 2:
            key = input("Enter key to update: ")
            if key in my_dict:
                new_value = input("Enter new value: ")
                my_dict[key] = new_value
            else:
                print("Key not found!")
        elif choice == 3:
            key = input("Enter key to delete: ")
            if key in my_dict:
                del my_dict[key]
            else:
                print("Key not found!")
        elif choice == 4:
            print("Dictionary:", my_dict)
        elif choice == 5:
            key = input("Enter key to search: ")
            if key in my_dict:
                print("Value:", my_dict[key])
            else:
                print("Key not found!")
        elif choice == 6:
            break
        else:
            print("Invalid choice!")


while True:
    choice = display_menu()
    if choice == 1:
        list_operations()
    elif choice == 2:
        tuple_operations()
    elif choice == 3:
        set_operations()
    elif choice == 4:
        dictionary_operations()
    elif choice == 5:
        print("Exiting...")
        break
    else:
        print("Invalid choice! Please try again.")
