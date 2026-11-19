"""Combined navigation and launcher client."""
import sys
from pi_common import main
if __name__ == '__main__':
    main(sys.argv[1:], profile=1, combined=True)
