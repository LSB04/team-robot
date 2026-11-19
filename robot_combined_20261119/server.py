"""Combined navigation and launcher server."""
import sys
from pc_common import main
if __name__ == '__main__':
    main(sys.argv[1:], profile=1, combined=True)
