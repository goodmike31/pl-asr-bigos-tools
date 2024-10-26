import sys
import configparser

def read_ini(section, key, filename='config.ini'):
    config = configparser.ConfigParser()
    config.read(filename)
    return config.get(section, key)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: read_ini.py <section> <key> <filename>")
        sys.exit(1)
    section = sys.argv[1]
    key = sys.argv[2]
    filename = sys.argv[3]
    print(read_ini(section, key, filename))
