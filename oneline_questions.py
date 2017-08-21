
from Utility import URLUtility
from bs4 import BeautifulSoup
import socket
socket.setdefaulttimeout(5)

def get_questions_from_sougou():
    fw = open('input/sougou.csv', 'w')

    for idx in range(1, 1000):
        print(idx)
        url = 'http://wenwen.sogou.com/cate/tag?tag_id=0&tp=0&pno=%d&ch=ww.fly.fy%d#questionList' % (idx, idx)
        succeed, content = URLUtility.get_response_content_for_url(url, 'utf8')
        if succeed:
            soup = BeautifulSoup(content)
            for elem in soup.select("ul.sort-lst a p"):
                fw.write(elem.text + '\n')
    fw.close()


get_questions_from_sougou()