import load_data
import load_model
import print_analogies
import load_analogies
import find_space
import find_space_kv
import adversarial_learning
import toxicity_scores

def main():
    load_data.main()
    client = load_model.main()
    print_analogies.main(client)
    analogies = load_analogies.main()
    indices, embed, gender_direction = find_space.main(client, analogies)
    adversarial_learning.main(client, analogies, indices, embed, gender_direction)
    print("Hello world!")

def main2():
    load_data.main()
    client = load_model.main()
    print_analogies.main(client)
    analogies = load_analogies.main()
    indices, embed, gender_direction = find_space.main(client, analogies)
    # adversarial_learning.main(client, analogies, indices, embed, gender_direction)
    print("Hello world!")

def main3():
    client = load_model.load_news_model()
    # toxicity_scores.main(client)
    print_analogies.main(client)
    analogies = load_analogies.main()
    indices, embed, gender_direction = find_space_kv.main(client, analogies)
    new_client = adversarial_learning.main(client, analogies, indices, embed, gender_direction)
    find_space_kv.race_scores(new_client, gender_direction, analogies)

if __name__ == "__main__": 
	main3()