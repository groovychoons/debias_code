
import os

ANALOGIES_FILE = os.path.join("data", "questions-words.txt")

def load_analogies(filename):
  """Loads analogies.

  Args:
    filename: the file containing the analogies.

  Returns:
    A list containing the analogies.
  """
  analogies = []
  with open(filename, "r") as fast_file:
    for line in fast_file:
      line = line.strip()
      # in the analogy file, comments start with :
      if line[0] == ":":
        continue
      words = line.split()
      # there are no misformatted lines in the analogy file, so this should
      # only happen once we're done reading all analogies.
      if len(words) != 4:
        print ("Invalid line: %s" % line)
        continue
      analogies.append(words)
  print ("loaded %d analogies" % len(analogies))
  return analogies

def main():
  analogies = load_analogies(ANALOGIES_FILE)
  print ("\n".join("%s is to %s as %s is to %s" % tuple(x) for x in analogies[:5]))
  return analogies
