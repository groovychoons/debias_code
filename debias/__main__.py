import load_data
import load_model
import print_analogies
import load_analogies
import find_space
import adversarial_learning

def main():
    load_data.main()
    client = load_model.main()
    print_analogies.main(client)
    analogies = load_analogies.main()
    indices, embed, gender_direction = find_space.main(client, analogies)
    adversarial_learning.main(client, analogies, indices, embed, gender_direction)
    print("Hello world!")

if __name__ == "__main__": 
	main()