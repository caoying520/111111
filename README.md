import pptx
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Cm, Pt, Inches
import os
import time
import requests
from bs4 import BeautifulSoup
import re
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import json
import copy
import six
from gooey import Gooey, GooeyParser
from szpscript.Broswer_script import Broswer
import win32api
import win32con
import zipfile
from szpscript.Uniprot_script import Uniprot
from szpscript.Public_Script import PublicScript
from win32com.client import Dispatch
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter
from Bio.Seq import Seq
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import docx
# from docx.shared import RGBColor
from szpscript.Spider_script import SpiderScript
from lxml import etree
import fitz
import sys
import traceback
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

"""
20230206: 往ppt插入质粒设计Word文档
20230228: 更新domain预测模型
20230307: 抓取商品商品信息
20230309: 更新质粒设计结构域算法
20230313: 更新下载文献模块
20230314: 更新文献解析模块
20230315: 增加位置限制
20230317: 优化下载
20230410: 解除电镜解析结构输出的限制
20230516: 修复分析文献Bug
20230517: 增加版本更新机制, 完善跨膜和AF2预测, 完善Biortus Plasmid Blast
20230518: 修复Sequence Analysis中的Bug
20230725: 修复浏览器版本Bug
20230921: 修复配体抓取方法
20260228：domain生成的方法更新
"""

'''
等待解决问题:
20230309: 输出5个商品蛋白信息优化(由于网页加载失败, 会导致输出小于5个)
'''


class Params:
    def __init__(self):
        pass

    # website = r"https://scilligence.net/Biortus/ELN/Login.aspx?https://scilligence.net/Biortus/ELN/Explorer.aspx"
    # selenium_waittime = 3000
    broswer_name = "Chrome"
    # chrome_driver_website = r"https://registry.npmmirror.com/binary.html?path=chromedriver/"
    chrome_driver_path = r"chromedriver.exe"
    proxies_port = None

    # 最新版本的驱动下载地址文件
    chrome_driver_latest_version_path = r"\\10.50.34.10\佰翱得共享文件\生物部\生物公共资料\8. 计算结构平台\2 公共软件\软件测试数据\DouRanELN\chrome_driver_version_dict.json"



class GetProteinCommercialInfo():
    __chrome_driver_path = Params.chrome_driver_path

    def __init__(self, uniprot_id, chrome_driver_path, work_dir):
        self.uniprot_id = uniprot_id
        self.chrome_driver_path = chrome_driver_path
        self.work_dir = os.path.join(work_dir, "commercial protein")

        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        # 初始化selenium 对象
        # self.bro = self.init_selenium()
        self.waittime = 300

        # 各个商品蛋白网站
        self.genecard_url = r"https://www.genecards.org/lookup/?text={}".format(uniprot_id)

        # 各个爬虫referer网站
        self.genecard_referer_url = r"https://www.genecards.org/"
        self.antibodyonline_referer_url = r"https://www.antibodies-online.com/"
        self.novus_referer_url = r"https://www.novusbio.com/"

        # 各个网站保存的位置
        self.genecard_html_file = os.path.join(work_dir, "genecard_{}.html".format(uniprot_id))
        self.sino_biological_html_file = os.path.join(work_dir, "sino_biological_{}.html".format(uniprot_id))
        self.antibodyonline_html_file = os.path.join(work_dir, "antibodyonline_{}.html".format(uniprot_id))
        self.novus_html_file = os.path.join(work_dir, "novus_{}.html".format(uniprot_id))

        # 等待赋值的变量
        self.pdfnum = 0

    def init_selenium(self):
        options = webdriver.ChromeOptions()

        # 自动下载pdf
        prefs = {'download.prompt_for_download': False,
                 'download.default_directory': self.work_dir,
                 "plugins.always_open_pdf_externally": True}

        options.add_experimental_option("prefs", prefs)
        try:
           service = Service(executable_path=self.chrome_driver_path)    #Selenium 新版本已经不推荐使用 chrome_options=options，应该改成 options=options
           bro = webdriver.Chrome(service=service,options=options)
        except:
           bro = webdriver.Chrome(options=options, executable_path=self.chrome_driver_path)

        bro.command_executor._commands["send_command"] = (
        "POST", '/session/$sessionId/chromium/send_command')

        params = {'cmd': 'Page.setDownloadBehavior',
                  'params': {'behavior': 'allow',
                             'downloadPath': self.work_dir}}

        bro.execute("send_command", params)

        return bro

    def request_genecard(self):
        if not os.path.exists(self.genecard_html_file):
            for times in range(10):
                try:
                    # print("  -->第{}尝试".format(times), flush=True)
                    download_status, html_text, response = SpiderScript.requests_text(self.genecard_url, self.genecard_referer_url, proxy_port=Params.proxies_port)

                    if download_status == 200:
                        with open(self.genecard_html_file, "w", encoding="utf-8") as f:
                            f.write(html_text)

                        break

                except:
                    time.sleep(5)

    def request_sino_biological(self):
        if os.path.exists(self.genecard_html_file) and not os.path.exists(self.sino_biological_html_file):

            with open(self.genecard_html_file, "r", encoding="utf-8") as f:
                entryid_html_content = f.read()

            html = etree.HTML(entryid_html_content)

            try:
                no_result = html.xpath('//*[@id="mobile-padding-wrapper"]/div[1]/div/main/h1')[0].text
                pass

            except:
                protein_product_list = html.xpath('//*[@id="RecombinantProteins"]/div')
                for protein_product in protein_product_list:
                    info_list = protein_product.xpath("./ul/li")
                    for info in info_list:
                        info_text = "".join(info.xpath('.//text()')).strip()
                        if re.search("^Sino Biological Recombinant Proteins", info_text):

                            url = r"https://www.genecards.org/" + info.xpath("./a/@href")[0]

                            # 让浏览器对指定url发起访问,访问NCBI蛋白质比对网址
                            self.bro = self.init_selenium()
                            self.bro.get(url)

                            # 等待加载完成
                            WebDriverWait(self.bro, self.waittime).until(
                                EC.presence_of_element_located(
                                    (By.XPATH,
                                     '//*[@class="label-item"]')))

                            try:
                                with open(self.sino_biological_html_file, "w", encoding="utf-8") as f:
                                    f.write(self.bro.page_source)
                            except:
                                pass

                            self.bro.close()
                            time.sleep(5)

    def download_sino_biological_datasheet(self):
        if os.path.exists(self.sino_biological_html_file):
            with open(self.sino_biological_html_file, "r", encoding="utf-8") as f:
                entryid_html_content = f.read()

            html = etree.HTML(entryid_html_content)

            coverlink_list = html.xpath('//*[@class="cover-link"]')
            label_item_list = []
            for coverlink in coverlink_list:
                # 判断对象是否未蛋白(非抗体类)
                coverlink_href_start = coverlink.get("href").split("/")[1]
                # print(coverlink_href_start)
                if "PROTEIN" in coverlink_href_start.upper():
                    label_item_list.extend(coverlink.xpath('.//*[@class="label-item"]'))

            # label_item_list = html.xpath('//*[@class="label-item"]')

            for label_item in label_item_list[:5]:
                pdf_url = r"https://cdn1.sinobiological.com/reagent/{}.pdf".format(label_item.text)
                pdf_name = os.path.basename(pdf_url)
                pdf_path = os.path.join(self.work_dir, pdf_name)
                print("下载{}".format(pdf_name), flush=True)
                if not os.path.exists(pdf_path):
                    self.bro = self.init_selenium()
                    self.bro.get(pdf_url)
                    # print(label_item.text, flush=True)
                    time.sleep(5)
                    self.bro.close()

    def download_antibodyonline_datasheet(self):
        if os.path.exists(self.genecard_html_file):

            with open(self.genecard_html_file, "r", encoding="utf-8") as f:
                entryid_html_content = f.read()

            html = etree.HTML(entryid_html_content)

            try:
                no_result = html.xpath('//*[@id="mobile-padding-wrapper"]/div[1]/div/main/h1')[0].text
                pass

            except:
                protein_product_list = html.xpath('//*[@id="RecombinantProteins"]/div')
                for protein_product in protein_product_list:
                    # 查找antibody online对应的信息
                    antibody_online_status = False
                    antibody_recommend_status = False

                    info_list = protein_product.xpath("./ul/li")
                    for info in info_list:
                        info_text = "".join(info.xpath('.//text()')).strip()
                        if re.search("^antibodies-online", info_text):
                            antibody_online_status = True

                        if antibody_online_status and re.search("^Recommended", info_text):
                            url_info_list = info.xpath(".//li")
                            for url_info in url_info_list[:5]:
                                url = r"https://www.genecards.org/" + url_info.xpath("./a/@href")[0]
                                url_protein_name = PublicScript.reformat_filename(url_info.xpath("./a")[0].text.strip())
                                url_html_path = self.antibodyonline_html_file.replace(".html", "_{}.html".format(url_protein_name))

                                # 访问url, 获取商品编号
                                html = None
                                if not os.path.exists(url_html_path):
                                    for times in range(10):
                                        try:
                                            # print("  -->第{}尝试".format(times), flush=True)

                                            status_code, html_txt, response = SpiderScript.requests_text(url, self.antibodyonline_referer_url,
                                                                                                      proxy_port=Params.proxies_port)

                                            if status_code == 200:
                                                with open(url_html_path, "w", encoding="utf-8") as f:
                                                    f.write(html_txt)

                                                html = etree.HTML(html_txt)

                                                break

                                        except:
                                            time.sleep(5)
                                else:
                                    with open(url_html_path, "r", encoding="utf-8") as f:
                                        entryid_html_content = f.read()

                                    html = etree.HTML(entryid_html_content)

                                # 判断网页是否加载正常
                                if html:
                                    pass
                                else:
                                    continue

                                catalognum = html.xpath('//*[@id="catalogue-id"]/strong')[0].text.replace("Catalog No.", "").strip()

                                # 下载pdf
                                pdf_url = r"https://www.antibodies-online.com/productsheets/{0}.pdf".format(catalognum)
                                pdf_name = os.path.basename(pdf_url)
                                pdf_path = os.path.join(self.work_dir, pdf_name)

                                print("下载{0}.pdf, url: {1}".format(catalognum, pdf_url), flush=True)
                                if not os.path.exists(pdf_path):
                                    for times in range(10):
                                        try:
                                            # print("  -->第{}尝试".format(times), flush=True)
                                            status_code, html_content, response = SpiderScript.requests_content(pdf_url,
                                                                                                                self.antibodyonline_referer_url,
                                                                                                                proxy_port=Params.proxies_port)

                                            if status_code == 200:
                                                with open(os.path.join(self.work_dir, "{}.pdf".format(catalognum)), "wb", ) as f:
                                                    f.write(html_content)

                                                break
                                        except:
                                            time.sleep(5)

                                    time.sleep(5)

    def download_novus_datasheet(self):
        if os.path.exists(self.genecard_html_file):

            with open(self.genecard_html_file, "r", encoding="utf-8") as f:
                entryid_html_content = f.read()

            html = etree.HTML(entryid_html_content)

            try:
                no_result = html.xpath('//*[@id="mobile-padding-wrapper"]/div[1]/div/main/h1')[0].text
                pass

            except:
                protein_product_list = html.xpath('//*[@id="RecombinantProteins"]/div')
                for protein_product in protein_product_list:
                    info_list = protein_product.xpath("./ul/li")
                    for info in info_list:
                        info_text = "".join(info.xpath('.//text()')).strip()
                        if re.search("^Novus Biologicals", info_text) and not re.search("^Novus Biologicals lysates", info_text):
                            # print(info_text, flush=True)
                            url = r"https://www.genecards.org/" + info.xpath("./a/@href")[0]

                            # 访问url
                            html = []
                            if not os.path.exists(self.novus_html_file):
                                for times in range(10):
                                    try:
                                        # print("  -->第{}尝试".format(times), flush=True)
                                        status_code, html_txt, response = SpiderScript.requests_text(url, self.novus_referer_url,
                                                                                                     proxy_port=Params.proxies_port)

                                        if status_code == 200:
                                            with open(self.novus_html_file, "w", encoding="utf-8") as f:
                                                f.write(html_txt)

                                            html = etree.HTML(html_txt)

                                            break
                                    except:
                                        time.sleep(5)
                            else:
                                with open(self.novus_html_file, "r", encoding="utf-8") as f:
                                    entryid_html_content = f.read()

                                html = etree.HTML(entryid_html_content)

                            # 判断网页是否加载正常
                            if html:
                                pass
                            else:
                                continue

                            protein_url_list = html.xpath('//*[@class="catalog_number_wrapper not3column"]//@href')
                            for protein_url in protein_url_list[:5]:
                                protein_url = r"https://www.novusbio.com" + protein_url

                                for times in range(10):
                                    try:
                                        # print("  -->第{}尝试".format(times), flush=True)
                                        status_code, html_txt, response = SpiderScript.requests_text(protein_url, self.novus_referer_url,
                                                                                                     proxy_port=Params.proxies_port)

                                        html = etree.HTML(html_txt)

                                        pdf_url = r"https:" + html.xpath('//*[@class="datasheet-download"]/@href')[0]
                                        pdf_name = os.path.basename(pdf_url)
                                        pdf_path = os.path.join(self.work_dir, pdf_name)

                                        break
                                    except:
                                        time.sleep(5)
                                else:
                                    # 网络故障
                                    pdf_url = None

                                if pdf_url:
                                    print("下载{0}, url: {1}".format(pdf_name, pdf_url), flush=True)
                                    for times in range(10):
                                        try:
                                            if not os.path.exists(pdf_path):
                                                status_code, html_content, response = SpiderScript.requests_content(pdf_url,
                                                                                                                    self.novus_referer_url,
                                                                                                                    proxy_port=Params.proxies_port)

                                                if status_code == 200:
                                                    with open(os.path.join(self.work_dir, pdf_name), "wb", ) as f:
                                                        f.write(html_content)

                                                    break
                                        except:
                                            time.sleep(5)

                                    time.sleep(5)
    def pdf2png(self):

        for roots, dirs, files in os.walk(self.work_dir):
            for file in files:
                if re.search("\.pdf$", file) and not re.search("^~", file):
                    # print(file)
                    pdf_path = os.path.join(roots, file)
                    pdf_name = file.replace(".pdf", "")
                    png_dir = os.path.join(roots, pdf_name)
                    output_file = os.path.join(png_dir, 'page_{}.png'.format(1))

                    if not os.path.exists(output_file):
                        if not os.path.exists(png_dir):
                            os.makedirs(png_dir)

                        pdf_file = fitz.open(pdf_path)

                        rotate = int(0)  # 设置图片的旋转角度为0
                        zoom_x = 2.0  # 设置图片相对于PDF文件在X轴上的缩放比例为2
                        zoom_y = 2.0  # 设置图片相对于PDF文件在Y轴上的缩放比例为2
                        trans = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)

                        pix = pdf_file[0].get_pixmap(matrix=trans, alpha=False)
                        pix.save(output_file)

                        # 关闭PDF文件
                        pdf_file.close()

    def getpdfnum(self):
        for roots, dirs, files in os.walk(self.work_dir):
            for file in files:
                if re.search("\.pdf$", file) and not re.search("^~", file):
                    self.pdfnum += 1

    def main(self):
        # 分析Genecard网站
        try:
            self.request_genecard()

        except Exception as e:
            print(e)
            print("genecard 读取失败")

        '''
        各个商品网站分析
        '''
        # 分析sino biological
        try:
            self.request_sino_biological()
        except Exception as e:
            print(e)
            print("sino biological 读取失败")
        try:
            self.download_sino_biological_datasheet()
        except Exception as e:
            print(e)
            print("sino biological 下载失败")

        # 分析antibody online
        try:
            self.download_antibodyonline_datasheet()
        except Exception as e:
            print(e)
            print("antibody online 下载失败")

        # 分析novus
        try:
            self.download_novus_datasheet()
        except Exception as e:
            print(e)
            print("novus 下载失败")

        # pdf2png
        self.pdf2png()

        # 计算Pdf的数量
        self.getpdfnum()


class GeneratePlasmidsWord:
    def __init__(self, protein_seq, protein_dna_seq, vector_name, tag_name, protein_name, save_dir, word_dir):

        self.protein_seq = protein_seq
        self.protein_dna_seq = protein_dna_seq

        self.vector_name, self.tag_name, self.protein_name = vector_name, tag_name, protein_name

        # self.specified_digest_seq = "ENLYFQ"
        self.specified_digest_seq = "QDSEVNQEAKPEVKPEVKPETHINLKVSDGSSEIFFKIKKTTPLRRL" \
                                    "MEAFAKRQGKEMDSLTFLYDGIEIQADQAPEDLDMEDNDIIEAHREQIGG"

        self.save_dir = save_dir
        self.word_dir = word_dir

        # 赋值
        self.protein_seq_8HisStrepII8HisSumo = "MHHHHHHHHWSHPQFEKHHHHHHHHQDSEVNQEAKPEVKPEVKPETHINLKVSDGSSEIFFKIKKTTPLRRL" \
                                               "MEAFAKRQGKEMDSLTFLYDGIEIQADQAPEDLDMEDNDIIEAHREQIGG"

        self.DNA_seq_8HisStrepII8HisSumo = "ATGCACCACCACCACCATCACCACCACTGGAGCCACCCCCAGTTCGAGAAGCACCACCACCACCATCACCACCACCAG" \
                                           "GACTCCGAGGTGAACCAGGAGGCCAAGCCCGAGGTGAAACCCGAAGTGAAGCCCGAGACCCACATCAACCTGAAGGTGTCC" \
                                           "GACGGCTCCTCCGAGATCTTCTTCAAGATCAAGAAGACCACCCCCCTGCGCCGCCTGATGGAGGCCTTCGCCAAGCGCCAG" \
                                           "GGCAAGGAGATGGACTCCCTGACCTTCCTGTACGACGGCATCGAGATCCAGGCCGACCAGGCCCCCGAGGACCTGGACATG" \
                                           "GAGGACAACGACATCATCGAGGCCCACCGCGAGCAGATCGGCGGC"

        self.protein_seq_8HisStrepIITEVGG = "MHHHHHHHHWSHPQFEKENLYFQGGG"

        self.DNA_seq_8HisStrepIITEVGG = "ATGCACCACCACCACCATCACCACCACTGGAGCCACCCGCAGTTCGAAAAGGAAAACCTGTATTTTCAGGGCGGCGGC"

        self.tag_info_dict = {
                                "His": {
                                    "seq": "",
                                    "color": "FFFF00"
                                },
                                "TEV": {
                                    "seq": "ENLYFQG",
                                    "color": "92D050",
                                    "digestion": "ENLYFQ"
                                },
                                "Sumo": {
                                    "seq": "QDSEVNQEAKPEVKPEVKPETHINLKVSDGSSEIFFKIKKTTPLRRLMEAFAKRQGKEMDSLTFLYDGIEIQADQAPEDLDMEDNDIIEAHREQIGG",
                                    "color": (
                                        112,
                                        48,
                                        160
                                    ),
                                    "digestion": ""
                                },
                                "Avi": {
                                    "seq": "GLNDIFEAQKIEWHE",
                                    "color": "FF00FF",
                                    "digestion": ""
                                },
                                "StrepII": {
                                    "seq": "WSHPQFEK",
                                    "color": "C0504D",
                                    "digestion": ""
                                },
                                "strepII": {
                                    "seq": "WSHPQFEK",
                                    "color": "C0504D",
                                    "digestion": ""
                                },
                                "strep": {
                                    "seq": "WSHPQFEK",
                                    "color": "C0504D",
                                    "digestion": ""
                                },
                                "Flag": {
                                    "seq": "DYKDDDDK",
                                    "color": "F79646",
                                    "digestion": ""
                                },
                                "3Flag": {
                                    "seq": "DYKDHDGDYKDHDIDYKDDDDK",
                                    "color": "F79646",
                                    "digestion": ""
                                },
                                "GST": {
                                    "seq": "SPILGYWKIKGLVQPTRLLLEYLEEKYEEHLYERDEGDKWRNKKFELGLEFPNLPYYIDGDVKLTQSMAIIRYIADKHNMLGGCPKERAEISMLEGAVLDIRYGVSRIAYSKDFETLKVDFLSKLPEMLKMFEDRLCHKTYLNGDHVTHPDFMLYDALDVVLYMDPMCLDAFPKLVCFKKRIEAIPQIDKYLKSSKYIAWPLQGWQATFGGGDHPPKSD",
                                    "color": (
                                        255,
                                        0,
                                        0
                                    ),
                                    "rgb-color": (
                                        255,
                                        0,
                                        0
                                    ),
                                    "digestion": ""
                                },
                                "MBP": {
                                    "seq": "KIEEGKLVIWINGDKGYNGLAEVGKKFEKDTGIKVTVEHPDKLEEKFPQVAATGDGPDIIFWAHDRFGGYAQSGLLAEITPDKAFQDKLYPFTWDAVRYNGKLIAYPIAVEALSLIYNKDLLPNPPKTWEEIPALDKELKAKGKSALMFNLQEPYFTWPLIAADGGYAFKYENGKYDIKDVGVDNAGAKAGLTFLVDLIKNKHMNADTDYSIAEAAFNKGETAMTINGPWAWSNIDTSKVNYGVTVLPTFKGQPSKPFVGVLSAGINAASPNKELAKEFLENYLLTDEGLEAVNKDKPLGAVALKSYEEELAKDPRIAATMENAQKGEIMPNIPQMSAFWYAVRTAVINAASGRQTVDEALKDAQTRITK",
                                    "color": (
                                        0,
                                        176,
                                        80
                                    ),
                                    "rgb-color": (
                                        0,
                                        176,
                                        80
                                    ),
                                    "digestion": ""
                                },
                                "TrxA": {
                                    "seq": "MSDKIIHLTD DSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANLA",
                                    "color": (
                                        227,
                                        108,
                                        9
                                    ),
                                    "digestion": ""
                                },
                                "IF2": {
                                    "seq": "TDVTIKTLAAERQTSVERLVQQFADAGIRKSADDSVSAQEKQTLIDHLNQKNSGPDKLTLQRKTRSTLNIPGTGGKSKSVQIEVRKKRTFVKRDPQEAERLAAEEQAQREAEEQARREAEESAKREAQQKAEREAAEQAKREAAEQAKREAAEKDKV",
                                    "color": (
                                        0,
                                        176,
                                        240
                                    ),
                                    "digestion": ""
                                },
                                "TF": {
                                    "seq": "MQVSVETTQGLGRRVTITIAADSIETAVKSELVNVAKKVRIDGFRKGKVPMNIVAQRYGASVRQDVLGDLMSRNFIDAIIKEKINPAGAPTYVPGEYKLGEDFTYSVEFEVYPEVELQGLEAIEVEKPIVEVTDADVDGMLDTLRKQQATWKEKDGAVEAEDRVTIDFTGSVDGEEFEGGKASDFVLAMGQGRMIPGFEDGIKGHKAGEEFTIDVTFPEEYHAENLKGKAAKFAINLKKVEERELPELTAEFIKRFGVEDGSVEGLRAEVRKNMERELKSAIRNRVKSQAIEGLVKANDIDVPAALIDSEIDVLRRQAAQRFGGNEKQALELPRELFEEQAKRRVVVGLLLGEVIRTNELKADEERVKGLIEEMASAYEDPKEVIEFYSKNKELMDNMRNVALEEQAVEAVLAKAKVTEKETTFNELMNQQASAG",
                                    "color": (
                                        255,
                                        192,
                                        0
                                    ),
                                    "digestion": ""
                                },
                                "NusA": {
                                    "seq": "NKEILAVVEAVSNEKALPREKIFEALESALATATKKKYEQEIDVRVQIDRKSGDFDTFRRWLVVDEVTQPTKEITLEAARYEDESLNLGDYVEDQIESVTFGRITTQTAKQVIVQKVREAERAMVVDQFREHEGEIITGVVKKVNRDNISLDLGNNAEAVILREDMLPRENFRPGDRVRGVLYSVRPEARGAQLFVTRSKPEMLIELFRIEVPEIGEEVIEIKAAARDPGSRAKIAVKTNDKRIDPVGACVGMRGARVQAVSTELGGERIDIVLWDDNPAQFVINAMAPADVASIVVDEDKHTMDIAVEAGNLAQAIGRNGQNVRLASQLSGWELNVMTVDDLQAKHQAEAHAAIDTFTKYLDIDEDFATVLVEEGFSTLEELAYVPMKELLEIEGLDEPTVEALRERAKNALATIAQAQEESLGDNKPADDLLNLEGVDRDLAFKLAARGVCTLEDLAEQGIDDLADIEGLTDEKAGALIMAARNICWFGDEA",
                                    "color": (
                                        33,
                                        88,
                                        104
                                    ),
                                    "digestion": ""
                                },
                                "SNAC": {
                                    "seq": "GSHHW",
                                    "color": "95B3D7",
                                    "digestion": ""
                                },
                                "Spytag": {
                                    "seq": "AHIVMVDAYKPTK",
                                    "color": "00B050",
                                    "digestion": ""
                                },
                                "sfGFP(S11)": {
                                    "seq": "RDHMVLHEYVNAAGIT",
                                    "color": "8064A2",
                                    "digestion": ""
                                },
                                "3C": {
                                    "seq": "LEVLFQGP",
                                    "color": "00B0F0",
                                    "digestion": ""
                                },
                                "HA\\(signal peptide\\)": {
                                    "seq": "MKTIIALSYIFCLVFA",
                                    "color": "D9D9D9",
                                    "digestion": ""
                                },
                                "GP67\\(signal peptide\\)": {
                                    "seq": "MLLVNQSHQGFNKEHTSKMVSAIVLYVLLAAAAHSAFA",
                                    "color": "D9D9D9",
                                    "digestion": ""
                                },
                                "pelB\\(signal peptide\\)": {
                                    "seq": "MKYLLPTAAAGLLLLAAQPAMA",
                                    "color": "D9D9D9",
                                    "digestion": ""
                                },
                                "ompA\\(signal peptide\\)": {
                                    "seq": "MKKTAIAIAVALAGFATVAQA",
                                    "color": "D9D9D9",
                                    "digestion": ""
                                },
                                "Gbeta": {
                                    "seq": "YKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
                                    "color": "548DD4",
                                    "digestion": ""
                                }
                            }

        self.protein_seq_font_size = 9
        self.protein_seq_font_name = "Courier New"

        self.plasmid_name = "{0}-{1}-{2}".format(self.vector_name, self.tag_name, self.protein_name)
        #self.full_protein_seq = self.protein_seq_8HisStrepII8HisSumo + self.protein_seq
        self.full_protein_seq = self.protein_seq_8HisStrepIITEVGG + self.protein_seq
        #self.full_dna_seq = self.DNA_seq_8HisStrepII8HisSumo + self.protein_dna_seq
        self.full_dna_seq = self.DNA_seq_8HisStrepIITEVGG + self.protein_dna_seq

    @staticmethod
    def calculate_protein_property(seq):
        # print(protein_seq)

        # initial protein seq
        protein_seq = str(seq).replace("*", "")
        seq = ProteinAnalysis(protein_seq)

        # protein length
        length = str(len(protein_seq)) + " aa"

        # protein molecular weight
        if len(protein_seq) == 0:
            molecular_weight = "/"
        else:
            molecular_weight = round(seq.molecular_weight(), 2)

        # protein isoelectric_point
        if len(protein_seq) == 0:
            isoelectric_point = "/"
        else:
            isoelectric_point = round(seq.isoelectric_point(), 2)

        # mole number (1 microgram)
        if len(protein_seq) == 0:
            mole_num = "/"
        else:
            mole_num = str(round(1 / molecular_weight * 10 ** 6, 2)) + " pMoles"

        # Molar Extinction coefficient
        if len(protein_seq) == 0:
            molar_extinction_coefficient_reduced = "/"
            molar_extinction_coefficient_disulfid = "/"
        else:
            molar_extinction_coefficient_reduced = seq.molar_extinction_coefficient()[0]
            molar_extinction_coefficient_disulfid = seq.molar_extinction_coefficient()[1]

        # A[280] of 1 mg/ml
        if len(protein_seq) == 0:
            absorb_reduced = "/"
            absorb_disulfid = "/"
        else:
            absorb_reduced = round(molar_extinction_coefficient_reduced / molecular_weight, 2)
            absorb_disulfid = round(molar_extinction_coefficient_disulfid / molecular_weight, 2)

        # 1 A[280] corr. to
        if len(protein_seq) == 0:
            reciprocal_absorb_reduced = "/"
            reciprocal_absorb_disulfid = "/"
        else:
            try:
                reciprocal_absorb_reduced = round(1 / absorb_reduced, 2)
                reciprocal_absorb_disulfid = round(1 / absorb_disulfid, 2)
            except:
                reciprocal_absorb_reduced = "/"
                reciprocal_absorb_disulfid = "/"

        # Charge at pH 7
        if len(protein_seq) == 0:
            charge_at_pH7 = "/"
        else:
            charge_at_pH7 = round(seq.charge_at_pH(7), 2)

        protein_property_dict = {"length": length,
                                 "molecular_weight": molecular_weight,
                                 "isoelectric_point": isoelectric_point,
                                 "mole_num": mole_num,
                                 "molar_extinction_coefficient_reduced": molar_extinction_coefficient_reduced,
                                 "absorb_reduced": absorb_reduced,
                                 "reciprocal_absorb_reduced": reciprocal_absorb_reduced,
                                 "charge_at_pH7": charge_at_pH7
                                }

        return protein_property_dict

    @staticmethod
    def get_tag_index(plasmidname, tag_info_dict, protein_seq, protein_seq_dict):

        protein_seq_no_tag = copy.deepcopy(str(protein_seq))

        # 匹配标签,标签位置,更改标签输出的颜色
        for tag_name, tag_info in tag_info_dict.items():
            # 匹配标签
            if tag_name == "His":
                tag_search_pattern = "-\d+{0}-|-\d+{0}".format(tag_name)
                if re.findall(tag_search_pattern, plasmidname):
                    for His_object in re.findall(tag_search_pattern, plasmidname):

                        His_num = re.search("\d+", His_object).group()

                        # 确定标签位置
                        protein_search_pattern = int(His_num) * "H"

                        # print(protein_search_pattern, str(protein_seq))

                        tag_start_index = re.search(protein_search_pattern, str(protein_seq_no_tag)).span()[0]
                        tag_end_index = re.search(protein_search_pattern, str(protein_seq_no_tag)).span()[1]

                        # 更改标签输出颜色
                        for protein_num, protein_info in protein_seq_dict.items():
                            if tag_start_index <= protein_num < tag_end_index:
                                protein_seq_dict[protein_num]["color"] = tag_info["color"]

                        # 生成不含标签的蛋白序列
                        protein_seq_no_tag = protein_seq_no_tag.replace(
                            protein_seq_no_tag[tag_start_index:tag_end_index], " " * (tag_end_index - tag_start_index), 1)

            else:
                # 匹配标签
                search_pattern = "-{0}-|-{0}".format(tag_name)
                # if tag_name == "TEV":
                #     search_pattern = "-{0}-|-{0}|-{1}-|-{1}".format(tag_name,"ENLYFQS")
                # else:
                #     search_pattern = "-{0}-|-{0}".format(tag_name)

                if re.search(search_pattern, plasmidname):
                    # input(tag_name)
                    # input(plasmidname)
                    # input(tag_info["seq"])
                    # input(str(protein_seq))

                    if tag_name == "TEV":
                        tag_search_pattern = "{0}|{1}".format(tag_info["seq"], "ENLYFQS")
                    else:
                        tag_search_pattern = tag_info["seq"]

                    # input(tag_search_pattern)

                    for Tag_object in re.finditer(tag_search_pattern, str(protein_seq_no_tag)):

                        # input(Tag_object)

                        # 确定标签位置
                        # tag_start_index = re.search(tag_info["seq"], str(protein_seq)).span()[0]
                        # tag_end_index = re.search(tag_info["seq"], str(protein_seq)).span()[1]
                        tag_start_index = Tag_object.span()[0]
                        tag_end_index = Tag_object.span()[1]

                        # 更改标签输出颜色
                        for protein_num, protein_info in protein_seq_dict.items():
                            if tag_start_index <= protein_num < tag_end_index:
                                protein_seq_dict[protein_num]["color"] = tag_info["color"]

                        # print(tag_name, tag_start_index, tag_end_index, protein_seq_dict)
                        # input(":")
                        #
                        # input("continue:")

                        protein_seq_no_tag = protein_seq_no_tag.replace(
                            protein_seq_no_tag[tag_start_index:tag_end_index],
                            " " * (tag_end_index - tag_start_index))

        return protein_seq_dict, protein_seq_no_tag

    @staticmethod
    def get_mutation_index(plasmidname, protein_seq, protein_seq_no_tag, protein_seq_dict, dna_seq_dict, specified_digest_seq):
        # 初始化突变氨基酸的位置
        mutation_symbol_update_dict = {}

        # get full protein symbol
        full_protein_pattern = "[GAVLIFWYDHNEKQMRSTCP]\d+-[GAVLIFWYDHNEKQMRSTCP]\d+"
        full_protein_symbol = re.findall(full_protein_pattern, plasmidname)

        # print(plasmidname, full_protein_symbol)

        try:
            full_protein_symbol_list = full_protein_symbol[0].split("-")
        except:
            return None, None, None

        # full_protein_symbol_dict
        full_protein_symbol_dict = {}
        full_protein_symbol_dict[re.search("\d+", full_protein_symbol_list[0]).group()] = re.search(
            "[GAVLIFWYDHNEKQMRSTCP]", full_protein_symbol_list[0]).group()
        full_protein_symbol_dict[re.search("\d+", full_protein_symbol_list[1]).group()] = re.search(
            "[GAVLIFWYDHNEKQMRSTCP]", full_protein_symbol_list[1]).group()

        # get full protein start or end index
        full_protein_start_index = int(re.search("\d+", full_protein_symbol_list[0]).group())
        full_protein_end_index = int(re.search("\d+", full_protein_symbol_list[1]).group())

        # get mutation symbol
        mutation_pattern = "[GAVLIFWYDHNEKQMRSTCP]\d+[GAVLIFWYDHNEKQMRSTCP]"
        mutation_symbol_list = re.findall(mutation_pattern, plasmidname)

        # mutation_symbol_dict
        mutation_symbol_dict = {}
        # print(mutation_symbol_list)
        for mutation in mutation_symbol_list:
            mutation_symbol_dict[re.search("\d+", mutation).group()] = mutation[-1]

        # get full mutaion protein symbol dict
        for full_protein_num in full_protein_symbol_dict.keys():
            if mutation_symbol_dict.get(full_protein_num):
                full_protein_symbol_dict[full_protein_num] = mutation_symbol_dict[full_protein_num]

        # calcalute protein(digested by TEV) property
        protein_digested_seq_list = str(protein_seq).split(specified_digest_seq)
        # 初始化,等待初始化为零
        protein_digested_seq = ""
        for i_protein_digested_seq in protein_digested_seq_list:
            protein_status, protein_seq_index = GeneratePlasmidsWord.find_protein_seq(i_protein_digested_seq, full_protein_symbol_dict,
                                                                 full_protein_end_index,
                                                                 full_protein_start_index)
            if protein_status:
                protein_digested_seq = i_protein_digested_seq

        protein_digested_property_dict = GeneratePlasmidsWord.calculate_protein_property(protein_digested_seq)

        # 找到蛋白序列
        temp_seq = ""
        temp_seq_length = 0
        # print(protein_seq_no_tag)
        for s in protein_seq_no_tag:
            if s.strip():
                temp_seq += s

            else:

                if len(temp_seq) >= full_protein_end_index - full_protein_start_index + 1:

                    # 判断temp seq是否为蛋白序列
                    protein_status, protein_seq_index = GeneratePlasmidsWord.find_protein_seq(temp_seq, full_protein_symbol_dict,
                                                                         full_protein_end_index,
                                                                         full_protein_start_index)

                    if protein_status:
                        # 得到蛋白初始氨基酸的位置
                        protein_aa_start_index = protein_seq_index + temp_seq_length

                        # 得到突变氨基酸的位置
                        mutation_symbol_update_dict = {}
                        # print(mutation_symbol_dict)
                        for mutation_index, mutation_seq in mutation_symbol_dict.items():
                            # print(mutation_index,type(mutation_index))
                            # print(protein_aa_start_index,type(protein_aa_start_index))
                            # input("wait:")
                            mutation_symbol_update_dict[
                                int(mutation_index) - full_protein_start_index + protein_aa_start_index] = mutation_seq

                        break

                # 计算蛋白序列前面的序列总长度
                if temp_seq.strip():
                    temp_seq_length += 1
                    temp_seq_length += len(temp_seq)
                else:
                    temp_seq_length += 1

                # temp seq 初始化
                temp_seq = ""

        else:
            if len(temp_seq) >= full_protein_end_index - full_protein_start_index + 1:
                # 判断temp seq是否为蛋白序列
                protein_status, protein_seq_index = GeneratePlasmidsWord.find_protein_seq(temp_seq, full_protein_symbol_dict,
                                                                     full_protein_end_index, full_protein_start_index)

                if protein_status:
                    # 得到蛋白初始氨基酸的位置
                    protein_aa_start_index = protein_seq_index + temp_seq_length

                    # 得到突变氨基酸的位置
                    mutation_symbol_update_dict = {}
                    # print(mutation_symbol_dict)
                    for mutation_index, mutation_seq in mutation_symbol_dict.items():
                        # print(mutation_index, type(mutation_index))
                        # print(protein_aa_start_index, type(protein_aa_start_index))
                        # print(full_protein_start_index)
                        # print(int(mutation_index) - full_protein_start_index + protein_aa_start_index)
                        # input("wait:")
                        mutation_symbol_update_dict[
                            int(mutation_index) - full_protein_start_index + protein_aa_start_index] = mutation_seq

        # 得到突变氨基酸的位置，及所在位置的颜色
        for mutaion_num in mutation_symbol_update_dict.keys():
            try:
                protein_seq_dict[int(mutaion_num)]["color"] = "FF0000"
            except:
                continue

        # 得到突变DNA的位置，及所在位置的颜色
        for protein_seq_num in protein_seq_dict.keys():
            dna_seq_dict[protein_seq_num * 3]["color"] = protein_seq_dict[protein_seq_num]["color"]
            dna_seq_dict[protein_seq_num * 3 + 1]["color"] = protein_seq_dict[protein_seq_num]["color"]
            dna_seq_dict[protein_seq_num * 3 + 2]["color"] = protein_seq_dict[protein_seq_num]["color"]

        return protein_seq_dict, dna_seq_dict, protein_digested_property_dict

    @staticmethod
    def find_protein_seq(seq, full_protein_symbol_dict, full_protein_end_index, full_protein_start_index):
        # print(seq)
        # print(full_protein_symbol_dict)
        # print(full_protein_end_index)
        # print(full_protein_start_index)

        for seq_index, s in enumerate(seq):
            if s == full_protein_symbol_dict[str(full_protein_start_index)]:
                # 如果没找到
                try:
                    if seq[seq_index + full_protein_end_index - full_protein_start_index] == full_protein_symbol_dict[
                        str(full_protein_end_index)]:
                        return True, seq_index
                except:
                    return False, False

        else:
            return False, False

    @staticmethod
    def seq_to_dict(seq, seq_type="protein"):
        seq_dict = {}
        for num, s in enumerate(seq):
            seq_dict[num] = {"seq": s, "color": ""}

        # 添加终止编码
        if seq_type == "protein":
            seq_dict[num + 1] = {"seq": "*", "color": ""}
        elif seq_type == "dna":
            seq_dict[num + 1] = {"seq": "T", "color": ""}
            seq_dict[num + 2] = {"seq": "A", "color": ""}
            seq_dict[num + 3] = {"seq": "A", "color": ""}

        return seq_dict

    @staticmethod
    def export_info_to_docx(plasmid_name, protein_seq_dict, dna_seq_dict, protein_property_dict,
                            protein_digested_property_dict, protein_seq_font_size, protein_seq_font_name, save_dir, word_dir):
        file = Document()

        # plasmid info
        file.add_heading("1.Plasmid:", level=1)
        file.add_paragraph("{0}".format(plasmid_name))

        # protein sequence info
        file.add_heading("2.Protein sequence:", level=1)
        for seq_index, seq_info_dict in protein_seq_dict.items():
            if (seq_index + 1) % 60 == 1:
                paragraph = file.add_paragraph()

                seq_num = seq_index + 1
                object = paragraph.add_run(str(seq_num))
                object.font.size = Pt(protein_seq_font_size)
                object.font.name = protein_seq_font_name

                blank_seq = (9 - len(str(seq_num))) * " "
                object = paragraph.add_run(blank_seq)
                object.font.size = Pt(protein_seq_font_size)
                object.font.name = protein_seq_font_name

                object = paragraph.add_run(seq_info_dict["seq"])
                object.font.size = Pt(protein_seq_font_size)
                object.font.name = protein_seq_font_name
                # if seq_info_dict[ 'color'] == "RED":
                #     object.font.color.rgb = WD_COLOR_INDEX.RED
                # elif seq_info_dict[ 'color'] == "GREEN":
                #     object.font.color.rgb = WD_COLOR_INDEX.GREEN
                # elif seq_info_dict[ 'color'] == "YELLOW":
                #     object.font.color.rgb = WD_COLOR_INDEX.YELLOW
                # if seq_info_dict['color']:
                #     # object.font.color.rgb = docx.shared.RGBColor(seq_info_dict['color'][0], seq_info_dict['color'][1],
                #     #                                        seq_info_dict['color'][2])
                #     set_highlight_color(object, seq_info_dict['color'])
                #

                if seq_info_dict['color']:
                    # GST,TEV标签颜色
                    if isinstance(seq_info_dict['color'], tuple):
                        object.font.color.rgb = docx.shared.RGBColor(seq_info_dict['color'][0], seq_info_dict['color'][1],
                                                         seq_info_dict['color'][2])
                    # object.font.color.rgb = docx.shared.RGBColor(seq_info_dict['color'][0], seq_info_dict['color'][1], seq_info_dict['color'][2])
                    else:
                        GeneratePlasmidsWord.set_highlight_color(object, seq_info_dict['color'])

            elif seq_index % 10 == 0:
                object = paragraph.add_run(" ")
                object.font.size = Pt(protein_seq_font_size)
                object.font.name = protein_seq_font_name

                object = paragraph.add_run(seq_info_dict["seq"])
                object.font.size = Pt(protein_seq_font_size)
                object.font.name = protein_seq_font_name
                # if seq_info_dict['color'] == "RED":
                #     object.font.color.rgb = WD_COLOR_INDEX.RED
                # elif seq_info_dict['color'] == "GREEN":
                #     object.font.color.rgb = WD_COLOR_INDEX.GREEN
                # elif seq_info_dict['color'] == "YELLOW":
                #     object.font.color.rgb = WD_COLOR_INDEX.YELLOW
                # if seq_info_dict['color']:
                #     # object.font.color.rgb = docx.shared.RGBColor(seq_info_dict['color'][0], seq_info_dict['color'][1],
                #     #                                        seq_info_dict['color'][2])
                #     set_highlight_color(object, seq_info_dict['color'])
                if seq_info_dict['color']:
                    # GST,TEV标签颜色
                    if isinstance(seq_info_dict['color'], tuple):
                        object.font.color.rgb = docx.shared.RGBColor(seq_info_dict['color'][0], seq_info_dict['color'][1],
                                                         seq_info_dict['color'][2])
                    # object.font.color.rgb = docx.shared.RGBColor(seq_info_dict['color'][0], seq_info_dict['color'][1], seq_info_dict['color'][2])
                    else:
                        GeneratePlasmidsWord.set_highlight_color(object, seq_info_dict['color'])


            else:
                object = paragraph.add_run(seq_info_dict["seq"])
                object.font.size = Pt(protein_seq_font_size)
                object.font.name = protein_seq_font_name
                # if seq_info_dict['color'] == "RED":
                #     object.font.color.rgb = WD_COLOR_INDEX.RED
                # elif seq_info_dict['color'] == "GREEN":
                #     object.font.color.rgb = WD_COLOR_INDEX.GREEN
                # elif seq_info_dict['color'] == "YELLOW":
                #     object.font.color.rgb = WD_COLOR_INDEX.YELLOW
                # if seq_info_dict['color']:
                #     # object.font.color.rgb = docx.shared.RGBColor(seq_info_dict['color'][0], seq_info_dict['color'][1],
                #     #                                        seq_info_dict['color'][2])
                #     set_highlight_color(object, seq_info_dict['color'])
                #
                if seq_info_dict['color']:
                    # GST,TEV标签颜色
                    if isinstance(seq_info_dict['color'], tuple):
                        object.font.color.rgb = docx.shared.RGBColor(seq_info_dict['color'][0], seq_info_dict['color'][1],
                                                         seq_info_dict['color'][2])
                    # object.font.color.rgb = docx.shared.RGBColor(seq_info_dict['color'][0], seq_info_dict['color'][1], seq_info_dict['color'][2])
                    else:
                        GeneratePlasmidsWord.set_highlight_color(object, seq_info_dict['color'])

        # dna sequence info
        file.add_heading("3.DNA sequence:", level=1)
        # print(dna_seq_dict)
        for seq_index, seq_info_dict in dna_seq_dict.items():

            if (seq_index + 1) % 60 == 1:

                paragraph = file.add_paragraph()

                seq_num = seq_index + 1
                object = paragraph.add_run(str(seq_num))
                object.font.size = Pt(protein_seq_font_size)
                object.font.name = protein_seq_font_name

                blank_seq = (9 - len(str(seq_num))) * " "
                object = paragraph.add_run(blank_seq)
                object.font.size = Pt(protein_seq_font_size)
                object.font.name = protein_seq_font_name

                # format_seq += s
                object = paragraph.add_run(seq_info_dict["seq"])
                object.font.size = Pt(protein_seq_font_size)
                object.font.name = protein_seq_font_name
                # if seq_info_dict[ 'color'] == "RED":
                #     object.font.color.rgb = WD_COLOR_INDEX.RED
                # elif seq_info_dict[ 'color'] == "GREEN":
                #     object.font.color.rgb = WD_COLOR_INDEX.GREEN
                # elif seq_info_dict[ 'color'] == "YELLOW":
                #     object.font.color.rgb = WD_COLOR_INDEX.YELLOW
                # if seq_info_dict['color']:
                #     # object.font.color.rgb = docx.shared.RGBColor(seq_info_dict['color'][0], seq_info_dict['color'][1],
                #     #                                        seq_info_dict['color'][2])
                #     set_highlight_color(object, seq_info_dict['color'])

                if seq_info_dict['color']:
                    # GST,TEV标签颜色
                    if isinstance(seq_info_dict['color'], tuple):
                        object.font.color.rgb = docx.shared.RGBColor(seq_info_dict['color'][0], seq_info_dict['color'][1],
                                                         seq_info_dict['color'][2])
                    # object.font.color.rgb = docx.shared.RGBColor(seq_info_dict['color'][0], seq_info_dict['color'][1], seq_info_dict['color'][2])
                    else:
                        GeneratePlasmidsWord.set_highlight_color(object, seq_info_dict['color'])

            elif seq_index % 10 == 0:
                object = paragraph.add_run(" ")
                object.font.size = Pt(protein_seq_font_size)
                object.font.name = protein_seq_font_name

                object = paragraph.add_run(seq_info_dict["seq"])
                object.font.size = Pt(protein_seq_font_size)
                object.font.name = protein_seq_font_name
                # if seq_info_dict['color'] == "RED":
                #     object.font.color.rgb = WD_COLOR_INDEX.RED
                # elif seq_info_dict['color'] == "GREEN":
                #     object.font.color.rgb = WD_COLOR_INDEX.GREEN
                # elif seq_info_dict['color'] == "YELLOW":
                #     object.font.color.rgb = WD_COLOR_INDEX.YELLOW
                # if seq_info_dict['color']:
                #     # object.font.color.rgb = docx.shared.RGBColor(seq_info_dict['color'][0], seq_info_dict['color'][1],
                #     #                                        seq_info_dict['color'][2])
                #     set_highlight_color(object, seq_info_dict['color'])
                #

                if seq_info_dict['color']:
                    # GST,TEV标签颜色
                    if isinstance(seq_info_dict['color'], tuple):
                        object.font.color.rgb = docx.shared.RGBColor(seq_info_dict['color'][0], seq_info_dict['color'][1],
                                                         seq_info_dict['color'][2])
                    # object.font.color.rgb = docx.shared.RGBColor(seq_info_dict['color'][0], seq_info_dict['color'][1], seq_info_dict['color'][2])
                    else:
                        GeneratePlasmidsWord.set_highlight_color(object, seq_info_dict['color'])

            else:
                object = paragraph.add_run(seq_info_dict["seq"])
                object.font.size = Pt(protein_seq_font_size)
                object.font.name = protein_seq_font_name
                # if seq_info_dict['color'] == "RED":
                #     object.font.color.rgb = WD_COLOR_INDEX.RED
                # elif seq_info_dict['color'] == "GREEN":
                #     object.font.color.rgb = WD_COLOR_INDEX.GREEN
                # elif seq_info_dict['color'] == "YELLOW":
                #     object.font.color.rgb = WD_COLOR_INDEX.YELLOW
                # object.font.color.rgb = eval("WD_COLOR_INDEX.{0}".format(seq_info_dict['color']))
                # object.font.color.rgb = docx.shared.RGBColor(250, 0, 0)
                if seq_info_dict['color']:
                    # GST,TEV标签颜色
                    if isinstance(seq_info_dict['color'], tuple):
                        object.font.color.rgb = docx.shared.RGBColor(seq_info_dict['color'][0], seq_info_dict['color'][1],
                                                         seq_info_dict['color'][2])
                    # object.font.color.rgb = docx.shared.RGBColor(seq_info_dict['color'][0], seq_info_dict['color'][1], seq_info_dict['color'][2])
                    else:
                        GeneratePlasmidsWord.set_highlight_color(object, seq_info_dict['color'])

        # protein property info
        file.add_heading("4.Estimated molecular characterization:", level=1)

        table = file.add_table(rows=9, cols=3)

        title_list = ["Analysis", "Entire Protein", "Protein digested by TEV"]
        for row in table.rows:
            for column_index, column in enumerate(row.cells):
                column.text = title_list[column_index]
                # 格式设置
                column.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                column.paragraphs[0].alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

                GeneratePlasmidsWord.set_cell_border(
                    column,
                    top={"sz": 12, "val": "single", "color": "#000000", "space": "0"},
                    bottom={"sz": 12, "val": "single", "color": "#000000", "space": "0"},
                    left={"sz": 12, "val": "single", "color": "#000000", "space": "0"},
                    right={"sz": 12, "val": "single", "color": "#000000", "space": "0"},
                )

        for column_index, column in enumerate(table.columns):
            for row_index, row in enumerate(column.cells[1:]):
                if column_index == 0:
                    row.text = list(protein_property_dict.items())[row_index][0]

                elif column_index == 1:
                    # print(list(protein_property_dict.items())[row_index])
                    row.text = str(list(protein_property_dict.items())[row_index][1])

                elif column_index == 2:
                    row.text = str(list(protein_digested_property_dict.items())[row_index][1])

                # 格式设置
                row.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                row.paragraphs[0].alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

                GeneratePlasmidsWord.set_cell_border(
                    row,
                    top={"sz": 12, "val": "single", "color": "#000000", "space": "0"},
                    bottom={"sz": 12, "val": "single", "color": "#000000", "space": "0"},
                    left={"sz": 12, "val": "single", "color": "#000000", "space": "0"},
                    right={"sz": 12, "val": "single", "color": "#000000", "space": "0"},
                )

        # 表对齐
        table.alignment = WD_TABLE_ALIGNMENT.CENTER

        # save file
        # try:
        savename = "{0}.docx".format(plasmid_name.replace("/", "").replace(":", "")).replace("*", "")
        word_save_dir = os.path.join(save_dir, word_dir)

        if not os.path.exists(word_save_dir):
            os.makedirs(word_save_dir)

        file.save(os.path.join(word_save_dir, savename))
        # file.close()
        print("word 文件保存路径: {0}".format(os.path.join(word_save_dir, savename)), flush=True)
        # file.save(r"docx5\{0}.docx".format("#" + plasmid_id + " " + plasmid_name.replace("/", "").replace(":", "")))
        # except:
        #     pass

    @staticmethod
    def set_cell_border(cell, **kwargs):
        """
        Set cell`s border
        Usage:
        set_cell_border(
            cell,
            top={"sz": 12, "val": "single", "color": "#FF0000", "space": "0"},
            bottom={"sz": 12, "color": "#00FF00", "val": "single"},
            left={"sz": 24, "val": "dashed", "shadow": "true"},
            right={"sz": 12, "val": "dashed"},
        )
        """
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()

        # check for tag existnace, if none found, then create one
        tcBorders = tcPr.first_child_found_in("w:tcBorders")
        if tcBorders is None:
            tcBorders = OxmlElement('w:tcBorders')
            tcPr.append(tcBorders)

        # list over all available tags
        for edge in ('left', 'top', 'right', 'bottom', 'insideH', 'insideV'):
            edge_data = kwargs.get(edge)
            if edge_data:
                tag = 'w:{}'.format(edge)

                # check for tag existnace, if none found, then create one
                element = tcBorders.find(qn(tag))
                if element is None:
                    element = OxmlElement(tag)
                    tcBorders.append(element)

                # looks like order of attributes is important
                for key in ["sz", "val", "color", "space", "shadow"]:
                    if key in edge_data:
                        element.set(qn('w:{}'.format(key)), str(edge_data[key]))

    @staticmethod
    def set_highlight_color(object, color_code):
        # Get the XML tag
        tag = object._r

        # Create XML element
        shd = OxmlElement('w:shd')

        # Add attributes to the element
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), color_code)

        tag.rPr.append(shd)

    def main(self):
        if self.protein_dna_seq:
            # calculate protein property
            # try:
            protein_property_dict = self.calculate_protein_property(self.full_protein_seq)
            # except Exception as e:
            #     print(e)
            #     pass

            # seq to dict
            protein_seq_dict = self.seq_to_dict(self.full_protein_seq)
            dna_seq_dict = self.seq_to_dict(self.full_dna_seq, seq_type="dna")

            # set tag index color
            # try:
            protein_seq_dict, protein_seq_no_tag = self.get_tag_index(self.plasmid_name, self.tag_info_dict, self.full_protein_seq, protein_seq_dict)
            # except Exception as e:
            #     print(e)
            #     pass

            # set mutation index color
            protein_seq_dict, dna_seq_dict, protein_digested_property_dict = self.get_mutation_index(self.plasmid_name,
                                                                                                     self.full_protein_seq,
                                                                                                     protein_seq_no_tag,
                                                                                                     protein_seq_dict,
                                                                                                     dna_seq_dict,
                                                                                                     self.specified_digest_seq)

            # print(protein_seq_dict, dna_seq_dict)
            if protein_seq_dict == None and dna_seq_dict == None:
                return

            self.export_info_to_docx(self.plasmid_name, protein_seq_dict, dna_seq_dict, protein_property_dict,
                                     protein_digested_property_dict, self.protein_seq_font_size, self.protein_seq_font_name,
                                     self.save_dir, self.word_dir)
        else:
            file = Document()

            # plasmid info
            file.add_paragraph("未找到模板")

            savename = "{0}.docx".format(self.plasmid_name.replace("/", "").replace(":", "")).replace("*", "")
            word_save_dir = os.path.join(self.save_dir, self.word_dir)

            if not os.path.exists(word_save_dir):
                os.makedirs(word_save_dir)

            file.save(os.path.join(word_save_dir, savename))
            # file.close()
            print("word 文件保存路径: {0}".format(os.path.join(word_save_dir, savename)), flush=True)


class Pae2Domain:
    _defaults = {
        'output_file': 'clusters.csv',
        'pae_power': 1.0,
        'pae_cutoff': 5.0,
        'resolution': 1.0,
        'library': 'igraph'
    }

    def __init__(self, pae_file, output_file=_defaults['output_file'], pae_power=_defaults['pae_power'], pae_cutoff=_defaults['pae_cutoff'],
                 resolution=_defaults['resolution'], library=_defaults['library']):
        self.pae_file = pae_file
        self.output_file = output_file
        self.pae_power = pae_power
        self.pae_cutoff = pae_cutoff
        self.resolution = resolution
        self.lib = library

    def main(self):
        pae = self.parse_pae_file()

        if self.lib == 'igraph':
            f = self.domains_from_pae_matrix_igraph
        else:
            f = self.domains_from_pae_matrix_networkx

        clusters = f(pae, pae_power=self.pae_power, pae_cutoff=self.pae_cutoff, graph_resolution=self.resolution)
        max_len = max([len(c) for c in clusters])
        clusters = [list(c) + [''] * (max_len - len(c)) for c in clusters]

        domain_list = []
        for c in clusters:
            domain_start = c[0] + 1
            domain_end = c[-1]
            for i in c:
                if i:
                    domain_end = i + 1
            domain = "{0}-{1}".format(domain_start, domain_end)
            domain_list.append(domain)

        return domain_list

        # output_file = self.output_file
        #
        # with open(output_file, 'wt') as outfile:
        #     for c in clusters:
        #         outfile.write(','.join([str(e) for e in c]) + '\n')

    def parse_pae_file(self):
        import json, numpy

        with open(self.pae_file, 'rt') as f:
            data = json.load(f)[0]

        if 'residue1' in data and 'distance' in data:
            # Legacy PAE format, keep for backwards compatibility.
            r1, d = data['residue1'], data['distance']
            size = max(r1)
            matrix = numpy.empty((size, size), dtype=numpy.float64)
            matrix.ravel()[:] = d
        elif 'predicted_aligned_error' in data:
            # New PAE format.
            matrix = numpy.array(data['predicted_aligned_error'], dtype=numpy.float64)
        else:
            raise ValueError('Invalid PAE JSON format.')

        return matrix

    def domains_from_pae_matrix_networkx(self, pae_matrix, pae_power=1, pae_cutoff=5, graph_resolution=1):
        '''
        Takes a predicted aligned error (PAE) matrix representing the predicted error in distances between each
        pair of residues in a model, and uses a graph-based community clustering algorithm to partition the model
        into approximately rigid groups.

        Arguments:

            * pae_matrix: a (n_residues x n_residues) numpy array. Diagonal elements should be set to some non-zero
              value to avoid divide-by-zero warnings
            * pae_power (optional, default=1): each edge in the graph will be weighted proportional to (1/pae**pae_power)
            * pae_cutoff (optional, default=5): graph edges will only be created for residue pairs with pae<pae_cutoff
            * graph_resolution (optional, default=1): regulates how aggressively the clustering algorithm is. Smaller values
              lead to larger clusters. Value should be larger than zero, and values larger than 5 are unlikely to be useful.

        Returns: a series of lists, where each list contains the indices of residues belonging to one cluster.
        '''
        try:
            import networkx as nx
        except ImportError:
            print(
                'ERROR: This method requires NetworkX (>=2.6.2) to be installed. Please install it using "pip install networkx" '
                'in a Python >=3.7 environment and try again.', flush=True)
            import sys
            sys.exit()
        import numpy
        weights = 1 / pae_matrix ** pae_power

        g = nx.Graph()
        size = weights.shape[0]
        g.add_nodes_from(range(size))
        edges = numpy.argwhere(pae_matrix < pae_cutoff)
        sel_weights = weights[edges.T[0], edges.T[1]]
        wedges = [(i, j, w) for (i, j), w in zip(edges, sel_weights)]
        g.add_weighted_edges_from(wedges)

        from networkx.algorithms import community

        clusters = community.greedy_modularity_communities(g, weight='weight', resolution=graph_resolution)
        return clusters

    def domains_from_pae_matrix_igraph(self, pae_matrix, pae_power=1, pae_cutoff=5, graph_resolution=1):
        '''
        Takes a predicted aligned error (PAE) matrix representing the predicted error in distances between each
        pair of residues in a model, and uses a graph-based community clustering algorithm to partition the model
        into approximately rigid groups.

        Arguments:

            * pae_matrix: a (n_residues x n_residues) numpy array. Diagonal elements should be set to some non-zero
              value to avoid divide-by-zero warnings
            * pae_power (optional, default=1): each edge in the graph will be weighted proportional to (1/pae**pae_power)
            * pae_cutoff (optional, default=5): graph edges will only be created for residue pairs with pae<pae_cutoff
            * graph_resolution (optional, default=1): regulates how aggressively the clustering algorithm is. Smaller values
              lead to larger clusters. Value should be larger than zero, and values larger than 5 are unlikely to be useful.

        Returns: a series of lists, where each list contains the indices of residues belonging to one cluster.
        '''
        try:
            import igraph
        except ImportError:
            print(
                'ERROR: This method requires python-igraph to be installed. Please install it using "pip install python-igraph" '
                'in a Python >=3.6 environment and try again.', flush=True)
            import sys
            sys.exit()
        import numpy
        weights = 1 / pae_matrix ** pae_power

        g = igraph.Graph()
        size = weights.shape[0]
        g.add_vertices(range(size))
        edges = numpy.argwhere(pae_matrix < pae_cutoff)
        sel_weights = weights[edges.T[0], edges.T[1]]
        g.add_edges(edges)
        g.es['weight'] = sel_weights

        vc = g.community_leiden(weights='weight', resolution_parameter=graph_resolution / 100, n_iterations=-1)
        membership = numpy.array(vc.membership)
        from collections import defaultdict
        clusters = defaultdict(list)
        for i, c in enumerate(membership):
            clusters[c].append(i)
        clusters = list(sorted(clusters.values(), key=lambda l: (len(l)), reverse=True))
        return clusters

    # if __name__ == '__main__':
    #     from time import time
    #     start_time = time()
    #     import argparse
    #     parser = argparse.ArgumentParser(description='Extract pseudo-rigid domains from an AlphaFold PAE matrix.')
    #     parser.add_argument('pae_file', type=str, help="Name of the PAE JSON file.")
    #     parser.add_argument('--output_file', type=str, default=_defaults['output_file'],
    #                         help=f'Name of output file (comma-delimited text format. Default: {_defaults["output_file"]}')
    #     parser.add_argument('--pae_power', type=float, default=_defaults['pae_power'],
    #                         help=f'Graph edges will be weighted as 1/pae**pae_power. Default: {_defaults["pae_power"]}')
    #     parser.add_argument('--pae_cutoff', type=float, default=_defaults['pae_cutoff'],
    #                         help=f'Graph edges will only be created for residue pairs with pae<pae_cutoff. Default: {_defaults["pae_cutoff"]}')
    #     parser.add_argument('--resolution', type=float, default=_defaults['resolution'],
    #                         help=f'Higher values lead to stricter (i.e. smaller) clusters. Default: {_defaults["resolution"]}')
    #     parser.add_argument('--library', type=str, default=_defaults['library'],
    #                         help=f'Graph library to use. "igraph" is about 40 times faster; "networkx" is pure Python. Default: {_defaults["library"]}')
    #     args = parser.parse_args()
    #     pae = parse_pae_file(args.pae_file)
    #     lib = args.library
    #     if lib == 'igraph':
    #         f = domains_from_pae_matrix_igraph
    #     else:
    #         f = domains_from_pae_matrix_networkx
    #     clusters = f(pae, pae_power=args.pae_power, pae_cutoff=args.pae_cutoff, graph_resolution=args.resolution)
    #     max_len = max([len(c) for c in clusters])
    #     clusters = [list(c) + [''] * (max_len - len(c)) for c in clusters]
    #     output_file = args.output_file
    #     with open(output_file, 'wt') as outfile:
    #         for c in clusters:
    #             outfile.write(','.join([str(e) for e in c]) + '\n')
    #     end_time = time()
    #     print(
    #         f'Wrote {len(clusters)} clusters to {output_file}. Biggest cluster contains {max_len} residues. Run time was {end_time-start_time:.2f} seconds.')
    #


class ProjectEvaluation:

    __chrome_driver_path = Params.chrome_driver_path

    def __init__(self, uniprot_id, person_name, email_address, template_ppt_path, chrome_driver_path, work_dir):
        # 判断使用位置
        self.location_limit()

        # 初始化参数
        self.uniprot_id = uniprot_id
        self.person_name = person_name
        self.email_address = email_address
        self.template_ppt_path = template_ppt_path
        self.chrome_driver_path = chrome_driver_path
        self.work_dir = work_dir
        self.word_dir = "word_{}".format(PublicScript.get_current_date())
        # read template ppt
        self.ppt_file_handle = Presentation(template_ppt_path)
        self.proxy = proxies

        self.domain_length_threhold = 0.1  # 结构域长度的阈值(大于总长度10%)
        self.plasmids_name_list = ["pET28a", "pFastBac", "Bacman", "pcDNA3.1"]  # 常用的质粒列表

        # 创建工作路径
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        # 待赋值
        self.uniprot_content_dict = {}
        self.protein_seq = ""
        self.gene_name = ""
        self.pptx_save_name = ""
        self.pptx_save_path = ""

        self.domain_list = []   # 结构域列表
        self.commercial_ppt_start_index = None

    def location_limit(self):
        if not os.path.exists(shared_version_file_path):
            win32api.MessageBox(0, "请在公司内部使用", "提醒", win32con.MB_TOPMOST)

            # 程序退出
            print("程序退出", flush=True)
            sys.exit()
        else:
            with open(shared_version_file_path, "r", encoding="utf-8") as f:
                version_dict = json.load(f)

            shared_version = version_dict["version"]
            soft_path = version_dict["soft_path"]

            if shared_version != software_version:
                shutil.copy(soft_path, os.path.join(os.getcwd(), os.path.basename(soft_path)))
                win32api.MessageBox(0, "点击确定自动下载最新版本, 请使用最新版本!!", "提醒", win32con.MB_TOPMOST)

                # 程序退出
                print("程序退出", flush=True)
                sys.exit()

    def get_af2_pic(self, af2_pdb_path):
        print("run get_af2_pic: ", af2_pdb_path, flush=True)

        url = "http://10.51.14.16/upload/af2_pic"

        pdb_file_name = os.path.basename(af2_pdb_path)

        with open(af2_pdb_path, 'rb') as pdb_file:
            files = {'file': (pdb_file_name, pdb_file, 'chemical/x-pdb')}
            response = requests.post(url, files=files)

        if response.status_code == 200:

            filename_list = response.json().get('filename', [])

            for filename in filename_list:
                obj = requests.get(f"http://10.51.14.16/af2_pic/{filename}")

                filename_path = os.path.join(self.work_dir, filename)

                if obj.status_code == 200:
                    with open(filename_path, 'wb') as f:
                        f.write(obj.content)

    def get_biortus_blast(self, seq_path):
        print("run get_biortus_blast: ", seq_path, flush=True)

        url = "http://10.51.14.16/upload/seq_blast"

        seq_file_name = os.path.basename(seq_path)

        with open(seq_path, 'rb') as pdb_file:
            files = {'file': (seq_file_name, pdb_file, 'application/fasta')}
            response = requests.post(url, files=files)

        # 打印响应内容
        if response.status_code == 200:

            filename_list = response.json().get('filename', [])

            for filename in filename_list:
                obj = requests.get("http://10.51.14.16/seq_blast/{}".format(filename))  #http://192.168.25.120/upload/seq_blast

                file_path = os.path.join(self.work_dir, filename)
                if obj.status_code == 200:
                    with open(file_path, "w", encoding="utf-8", newline='') as f:
                        f.write(obj.text)

    def get_uniprot_xml(self, uniprotid):
        print("run get_uniprot_xml: ", uniprotid, flush=True)
        obj = requests.get("http://10.51.14.16/xml/{}.xml".format(uniprotid))

        if obj.status_code == 200:
            file_path = os.path.join(self.work_dir, uniprotid + ".xml")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(obj.text)


    def get_uniprot_html_content(self):
        '''

        :param unipro_id,proxies:
        :return: html_content txt
        '''

        if not os.path.exists(os.path.join(self.work_dir, self.uniprot_id + ".xml")):
            self.get_uniprot_xml(self.uniprot_id)

            if not os.path.exists(os.path.join(self.work_dir, self.uniprot_id + ".xml")):
                status_code, xml_file_content = Uniprot.download_xml(self.uniprot_id, proxy_port=Params.proxies_port)

                # save uniprot html to file
                with open(os.path.join(self.work_dir, self.uniprot_id + ".xml"), "w", encoding="utf-8") as f:
                    f.write(xml_file_content)

    def process_uniprot_html(self):
        '''

        :param uniprot_id:
        :return: uniprot content dict
        '''

        # initial uniprot content dict
        uniprot_content_dict = {}

       
        xml_info_dict = Uniprot.parse_xml(os.path.join(self.work_dir, self.uniprot_id + ".xml"))
        # with open(os.path.join(self.work_dir, self.uniprot_id + ".json"), "w", encoding="utf-8") as f:
        #     json.dump(xml_info_dict, f, ensure_ascii=False, indent=4)
        
        uniprot_content_dict["protein_recommended_name"] = xml_info_dict['proteiname_full']
        uniprot_content_dict["protein_short_name"] = xml_info_dict['proteinname_short']
        uniprot_content_dict["protein_gene_name"] = xml_info_dict['target_name']
        uniprot_content_dict["protein_organism_name"] = xml_info_dict['scientific_organism']
        uniprot_content_dict["cellular_component"] = ", ".join(xml_info_dict["subcellular_location"])
        # uniprot_reference_sequence
        try:
            uniprot_reference_sequence = "https://www.uniprot.org/uniprot/{}".format(self.uniprot_id)
            uniprot_content_dict["uniprot_reference_sequence"] = uniprot_reference_sequence
        except:
            uniprot_content_dict["uniprot_reference_sequence"] = ""

        # sequence
        sequence = xml_info_dict['sequence']
        uniprot_content_dict["protein_sequence_list"] = []
        if sequence:
            temp = ""
            for i in sequence:
                temp += i
                if len(temp) == 10:
                    uniprot_content_dict["protein_sequence_list"].append(temp)
                    temp = ""

            if temp:
                uniprot_content_dict["protein_sequence_list"].append(temp)

        # function
        uniprot_content_dict["protein_function_list"] = xml_info_dict['function'].split(".")

        # PDB Sum
        uniprot_content_dict["pdb_summary_list"] = xml_info_dict['PDBsum_list']
        print("uniprot_content_dict['protein_sequence_list']", uniprot_content_dict["pdb_summary_list"], flush=True)

        # 赋值
        self.uniprot_content_dict = uniprot_content_dict
        self.protein_seq = "".join(uniprot_content_dict["protein_sequence_list"])
        self.gene_name = uniprot_content_dict["protein_gene_name"]

    def ppt_cover(self):
        file = self.ppt_file_handle
        project_evaluation_name_short = self.gene_name

        # add slide 1, layout=0
        slice = file.slides.add_slide(file.slide_layouts[0])

        # 13,12 = placeholder index (titile,time)
        project_evaluation_title = slice.placeholders[0]
        project_evaluation_time = slice.placeholders[19]

        project_evaluation_title.text = "Project Evaluation: " + project_evaluation_name_short
        project_evaluation_time.text = time.strftime("%Y-%m-%d", time.localtime())

        # project_evaluation_time.font.superscript = True

    def ppt_protein_sequence_analysis(self, rows=5, columns=4):
        file = self.ppt_file_handle
        # add slide 1, layout=1
        slice = file.slides.add_slide(file.slide_layouts[1])

        # 14,13,12 = placeholder (protein info,protein sequence,protein table)
        pptx_protein_info = slice.placeholders[14]
        pptx_protein_sequence = slice.placeholders[13]
        pptx_protein_table = slice.placeholders[12]

        pptx_protein_title= slice.placeholders[18]
        title = pptx_protein_title.text_frame
        title_paragraph = title.add_paragraph()
        title_run = title.paragraphs[0].add_run()
        title_run.text = "Protein Sequence Analysis"
        title_run.font.size = Pt(28)
        title_run.font.bold = True
        title_run.font.color.rgb = RGBColor(146, 50, 34)
        title_run.font.name = "Times New Roman"

        # protein info
        protein_info_text_frame = pptx_protein_info.text_frame
        protein_info_paragraph = protein_info_text_frame.add_paragraph()
        ## protein name
        protein_name = protein_info_text_frame.paragraphs[0].add_run()
        protein_name.text = "Protein name: "
        protein_name.font.bold = True

        protein_name_info = protein_info_text_frame.paragraphs[0].add_run()
        protein_name_info.text = self.uniprot_content_dict["protein_recommended_name"]

        ## protein short name
        protein_short_name = protein_info_paragraph.add_run()
        protein_short_name.text = "Short name: "
        protein_short_name.font.bold = True

        protein_short_name_info = protein_info_paragraph.add_run()
        protein_short_name_info.text = self.uniprot_content_dict["protein_short_name"] + "\n"

        ## protein organism
        protein_organism = protein_info_paragraph.add_run()
        protein_organism.text = "Organism: "
        protein_organism.font.bold = True

        protein_organism_info = protein_info_paragraph.add_run()
        protein_organism_info.text = self.uniprot_content_dict["protein_organism_name"] + "\n"

        ## protein cellular_component
        protein_cellular_componet = protein_info_paragraph.add_run()
        protein_cellular_componet.text = "Cellular component: "
        protein_cellular_componet.font.bold = True

        protein_cellular_componet.info = protein_info_paragraph.add_run()
        protein_cellular_componet.info.text = self.uniprot_content_dict["cellular_component"] + "\n"

        ## protein uniprot_reference_sequence
        protein_uniprot_reference_seruence = protein_info_paragraph.add_run()
        protein_uniprot_reference_seruence.text = "Uniprot Reference Sequence: "
        protein_uniprot_reference_seruence.font.bold = True

        protein_uniprot_reference_seruence_info = protein_info_paragraph.add_run()
        protein_uniprot_reference_seruence_info.text = self.uniprot_content_dict["uniprot_reference_sequence"] + "\n"
        #### 超链接
        hlink = protein_uniprot_reference_seruence_info.hyperlink
        hlink.address = self.uniprot_content_dict["uniprot_reference_sequence"]

        ## protein sequence header
        protein_sequence_header = protein_info_paragraph.add_run()
        protein_sequence_header.text = "{}(full length) sequence:".format(
            self.uniprot_content_dict["protein_short_name"])
        protein_sequence_header.font.bold = True

        # protein sequence
        uniprot_protein_sequence = ""
        # initial protein_sequence length
        total_protein_sequence_length = 0
        # print(uniprot_content_dict["protein_sequence_list"])
        for protein_sequence in self.uniprot_content_dict["protein_sequence_list"]:
            total_protein_sequence_length += len(protein_sequence)

            # if (total_protein_sequence_length - 10) % 60 == 0:
            #     if (total_protein_sequence_length - 10) / 60 != 0:
            #         uniprot_protein_sequence += "\n" + " " * (4 - len(str((total_protein_sequence_length - 10) + 1))) + str((total_protein_sequence_length - 10) + 1)
            #     elif (total_protein_sequence_length - 10) / 60 == 0:
            #         uniprot_protein_sequence += " " * 3 + str((total_protein_sequence_length - 10) + 1)

            if 0 < (total_protein_sequence_length) % 60 <= 10:
                if (total_protein_sequence_length - 10) % 60 == 0:
                    if (total_protein_sequence_length - 10) / 60 != 0:
                        uniprot_protein_sequence += "\n" + " " * (
                                    4 - len(str((total_protein_sequence_length - 10) + 1))) + str(
                            (total_protein_sequence_length - 10) + 1)
                    elif (total_protein_sequence_length - 10) / 60 == 0:
                        uniprot_protein_sequence += " " * 3 + str((total_protein_sequence_length - 10) + 1)
                else:
                    if total_protein_sequence_length <= 10:
                        uniprot_protein_sequence += " " * 3 + str(1)
                    elif total_protein_sequence_length > 10:
                        uniprot_protein_sequence += "\n" + " " * (4 - len(str(total_protein_sequence_length))) + str(
                            (60 * int(total_protein_sequence_length / 60)) + 1)

            uniprot_protein_sequence += " " + protein_sequence
        print('111111')
        try:
            pptx_protein_sequence_text_frame = pptx_protein_sequence.text_frame
            pptx_protein_sequence_text = pptx_protein_sequence_text_frame.paragraphs[0].add_run()

            pptx_protein_sequence_text.text = uniprot_protein_sequence

            # 序列字体
            if 12 * 11 / uniprot_protein_sequence.count("\n") > 14:
                sequence_font_size = 14

            else:
                sequence_font_size = 12 * 11 / uniprot_protein_sequence.count("\n")

            if sequence_font_size < 1:
                sequence_font_size = 1

            pptx_protein_sequence_text.font.size = Pt(sequence_font_size)

            # protein table
            table = pptx_protein_table.insert_table(rows, columns).table

            # row height
            for row in table.rows:
                row.height = 150000
                # print(row.height)
            ## protein table header
            table.cell(0, 0).text, table.cell(0, 1).text, table.cell(0, 2).text, table.cell(0,
                                                                                            3).text = "Items", "Results", "Items", "Results"
            # table.cell(0, 0).text_frame.paragraphs.font.size = Pt(14)
        except:
            pass
        print('22222')
        try:
            ## protein table info
            protein_property_dict = self.calculate_protein_property("".join(self.uniprot_content_dict["protein_sequence_list"]))
            table.cell(1, 0).text, table.cell(2, 0).text, table.cell(3, 0).text, table.cell(4,
                                                                                            0).text = "Length", "Molecular weight", "1 microgram=", "Instability index"
            table.cell(1, 1).text, table.cell(2, 1).text, table.cell(3, 1).text, table.cell(4, 1).text = \
            protein_property_dict["length"], str(protein_property_dict["molecular_weight"]), protein_property_dict[
                "mole_num"], str(protein_property_dict["instability_index"])
            table.cell(1, 2).text, table.cell(2, 2).text, table.cell(3, 2).text, table.cell(4,
                                                                                            2).text = "Cys(C)", "1A[280] corr. to", "Isoelectric Point", ""
            table.cell(1, 3).text, table.cell(2, 3).text, table.cell(3, 3).text = str(
                "".join(self.uniprot_content_dict["protein_sequence_list"]).count("C")), protein_property_dict[
                                                                                      "reciprocal_absorb_reduced"], str(
                protein_property_dict["isoelectric_point"])

            ## protein table format
            for cell in table.iter_cells():
                for paragraph in cell.text_frame.paragraphs:
                    paragraph.font.size = Pt(12)
                    # paragraph.line_spacing = 1
                    paragraph.font.name = "Times New Roman"
                    paragraph.alignment = PP_ALIGN.CENTER
                    paragraph.font.color.rgb = RGBColor(0, 0, 0)

            # 设置表格颜色
            # 合并单元格
            table.cell(4, 2).merge(table.cell(4, 3))

            if protein_property_dict["instability_index"] <= 40:
                # para = table.cell(4, 2).text_frame.add_paragraph()
                para = table.cell(4, 2).text_frame.paragraphs[0]
                para.text = "This classifies the protein as stable"
                para.font.color.rgb = RGBColor(255, 0, 0)
                para.font.size = Pt(12)
                para.font.name = "Times New Roman"
                para.alignment = PP_ALIGN.CENTER
                # table.cell(4, 2).text = "This classifies the protein as stable"
            else:
                # para = table.cell(4, 2).text_frame.add_paragraph()
                para = table.cell(4, 2).text_frame.paragraphs[0]
                para.text = "This classifies the protein as unstable"
                para.font.color.rgb = RGBColor(255, 0, 0)
                para.font.size = Pt(12)
                para.font.name = "Times New Roman"
                para.alignment = PP_ALIGN.CENTER
                # table.cell(4, 2).text = "This classifies the protein as unstable"
            # table.cell(4, 2).text_frame.add_paragraph().font.color.rgb = RGBColor(255, 0, 0)
        except:
            pass
    def ppt_protein_infomation(self):
        file = self.ppt_file_handle

        # add slide 1, layout=2
        slice = file.slides.add_slide(file.slide_layouts[2])

        # 10 = placeholder index (function)
        pptx_protein_title = slice.placeholders[18]
        title = pptx_protein_title.text_frame
        title_paragraph = title.add_paragraph()
        title_run = title.paragraphs[0].add_run()
        title_run.text = "Protein Information"
        title_run.font.size = Pt(28)
        title_run.font.bold = True
        title_run.font.color.rgb = RGBColor(146, 50, 34)
        title_run.font.name = "Times New Roman"

        pptx_protein_function = slice.placeholders[10]

        # protein function
        pptx_protein_function.text = ""
        for protein_function in self.uniprot_content_dict["protein_function_list"]:
            pptx_protein_function.text += protein_function.strip() + ".\n\n"

            if len(pptx_protein_function.text) > 600:
                break
    def ppt_compounds_info(self, rows=7, columns=4):
        ppt_file = self.ppt_file_handle

        slice = ppt_file.slides.add_slide(ppt_file.slide_layouts[15])

        pptx_protein_title = slice.placeholders[18]
        title = pptx_protein_title.text_frame
        title_paragraph = title.add_paragraph()
        title_run = title.paragraphs[0].add_run()
        title_run.text = "Compounds Information"
        title_run.font.size = Pt(28)
        title_run.font.bold = True
        title_run.font.color.rgb = RGBColor(146, 50, 34)
        title_run.font.name = "Times New Roman"


        ppt_pdb_summary_table = slice.placeholders[12]

        # protein table
        table = ppt_pdb_summary_table.insert_table(rows, columns).table

        # row height
        for row in table.rows:
            row.height = 150000
            # print(row.height)

        ## protein table header
        table.cell(0, 0).text, table.cell(0, 1).text, table.cell(0, 2).text, table.cell(0, 3).text = \
            "Distributor", "Product Name", "Effect", "Compound"

        ## protein table format
        for cell in table.iter_cells():
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(12)
                # paragraph.line_spacing = 1
                paragraph.font.name = "Times New Roman"
                paragraph.alignment = PP_ALIGN.CENTER
                paragraph.font.color.rgb = RGBColor(0, 0, 0)

    def ppt_topology_prediction(self, defalut_download_path="", waittime=60):
        file = self.ppt_file_handle

        if not os.path.exists(os.path.join(self.work_dir, 'topology_prediction.png')):
            try:
                # topology_prediction_png_url = "http://wlab.ethz.ch/protter/create?seq={0}&tm=auto&mc=lightsalmon&lc=blue&tml=numcount&numbers&legend&n:signal%20peptide,cc:white,fc:red,bc:red=Phobius.SP&n:N-glyco%20motif,s:box,fc:forestgreen,bc:forestgreen=(N).[ST]&format=png".format(protein_seq)
                # topology_prediction_png_url = "http://wlab.ethz.ch/protter/#seq={0}&tm=auto&mc=lightsalmon&lc=blue&tml=numcount&numbers&legend&n:signal%20peptide,cc:white,fc:red,bc:red=Phobius.SP&n:N-glyco%20motif,s:box,fc:forestgreen,bc:forestgreen=(N).[ST]&format=png".format(protein_seq)
                topology_prediction_png_url = "http://wlab.ethz.ch/protter/create?seq={0}&tm=auto&mc=lightsalmon&lc=blue&tml=numcount&numbers&legend&n:signal%20peptide,cc:white,fc:red,bc:red=Phobius.SP&n:N-glyco%20motif,s:box,fc:forestgreen,bc:forestgreen=(N).[ST]&format=png".format(
                    self.protein_seq)
                print("跨膜预测url: ", topology_prediction_png_url, flush=True)

                for times in range(10):
                    try:
                        topology_prediction_png = requests.get(topology_prediction_png_url,
                                                               proxies=None,
                                                               timeout=waittime)

                        if topology_prediction_png.status_code == 200:
                            with open(os.path.join(self.work_dir, 'topology_prediction.png'), 'wb') as f:
                                f.write(topology_prediction_png.content)

                            break
                    except:
                        time.sleep(5)

                # add slide 1, layout=2

                if os.path.exists(os.path.join(self.work_dir, 'topology_prediction.png')):
                    slice = file.slides.add_slide(file.slide_layouts[5])

                    pptx_protein_title = slice.placeholders[18]
                    title = pptx_protein_title.text_frame
                    title_paragraph = title.add_paragraph()
                    title_run = title.paragraphs[0].add_run()
                    title_run.text = "Topology Prediction"
                    title_run.font.size = Pt(28)
                    title_run.font.bold = True
                    title_run.font.color.rgb = RGBColor(146, 50, 34)
                    title_run.font.name = "Times New Roman"

                    left = Cm(1.9)
                    top = Cm(3.9)
                    # height = Cm(15)
                    width = Cm(23)
                    pic = slice.shapes.add_picture(os.path.join(self.work_dir, r'topology_prediction.png'), left, top,
                                                   width=width)

                    # 图片置于文本框下方
                    # slice.shapes._spTree.insert(1,pic._element)

                    pptx_trans_membrane_helices_prediction_text = slice.placeholders[10]

                    pptx_trans_membrane_helices_prediction_text.text = "Based on the topological structure and function, this protein may be an/a ****** protein."

                else:
                    slice = file.slides.add_slide(file.slide_layouts[5])
                    pptx_protein_title = slice.placeholders[18]
                    title = pptx_protein_title.text_frame
                    title_paragraph = title.add_paragraph()
                    title_run = title.paragraphs[0].add_run()
                    title_run.text = "Topology Prediction"
                    title_run.font.size = Pt(28)
                    title_run.font.bold = True
                    title_run.font.color.rgb = RGBColor(146, 50, 34)
                    title_run.font.name = "Times New Roman"


            except Exception as e:
                print(e)
                slice = file.slides.add_slide(file.slide_layouts[5])
                pptx_protein_title = slice.placeholders[18]
                title = pptx_protein_title.text_frame
                title_paragraph = title.add_paragraph()
                title_run = title.paragraphs[0].add_run()
                title_run.text = "Topology Prediction"
                title_run.font.size = Pt(28)
                title_run.font.bold = True
                title_run.font.color.rgb = RGBColor(146, 50, 34)
                title_run.font.name = "Times New Roman"


        else:

            # add slide 1, layout=2
            try:
                slice = file.slides.add_slide(file.slide_layouts[5])
                pptx_protein_title = slice.placeholders[18]
                title = pptx_protein_title.text_frame
                title_paragraph = title.add_paragraph()
                title_run = title.paragraphs[0].add_run()
                title_run.text = "Topology Prediction"
                title_run.font.size = Pt(28)
                title_run.font.bold = True
                title_run.font.color.rgb = RGBColor(146, 50, 34)
                title_run.font.name = "Times New Roman"

                left = Cm(1.9)
                top = Cm(3.9)
                # height = Cm(15)
                width = Cm(23)
                pic = slice.shapes.add_picture(os.path.join(self.work_dir, r'topology_prediction.png'), left, top,
                                               width=width)

                # 图片置于文本框下方
                # slice.shapes._spTree.insert(1,pic._element)

                pptx_trans_membrane_helices_prediction_text = slice.placeholders[10]

                pptx_trans_membrane_helices_prediction_text.text = "Based on the topological structure and function, this protein may be an/a ****** protein."

            except Exception as e:
                print(e)
                slice = file.slides.add_slide(file.slide_layouts[5])
                pptx_protein_title = slice.placeholders[18]
                title = pptx_protein_title.text_frame
                title_paragraph = title.add_paragraph()
                title_run = title.paragraphs[0].add_run()
                title_run.text = "Topology Prediction"
                title_run.font.size = Pt(28)
                title_run.font.bold = True
                title_run.font.color.rgb = RGBColor(146, 50, 34)
                title_run.font.name = "Times New Roman"

    def ppt_alphafold2_structure_prediction(self):
        ppt_file = self.ppt_file_handle

        AF2_PDB_name = "AF2-{}.pdb".format(self.uniprot_id)
        AF2_url = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v6.pdb".format(self.uniprot_id)
        print("AF2_url: ", AF2_url, flush=True)
        # 判断是否下载过
        print(os.path.exists(os.path.join(self.work_dir, AF2_PDB_name)))
        if not os.path.exists(os.path.join(self.work_dir, AF2_PDB_name)):
            print("下载AF2 pdb文件", flush=True)
            for times in range(10):
                try:
                    # 获取pdb
                    response = requests.get(
                        url=AF2_url,
                        #proxies=proxies,#使用vpn访问
                        proxies=None,#不使用vpn，本地ip访问
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
                            "Referer": "https://alphafold.ebi.ac.uk/entry/{}".format(self.uniprot_id)
                        }
                    )
                    print('------------')
                    print("AF2 response.status_code: ", response.status_code, flush=True)
                    if response.status_code == 200:
                        with open(os.path.join(self.work_dir, AF2_PDB_name), "w", encoding="utf-8") as f:
                            f.write(response.text)

                        break
                except Exception as e:
                    print("下载AF2 pdb文件失败: ", e, flush=True)
                    time.sleep(3)
        # 上传至服务器, 生成结构图片
        AF2_PDB_path = os.path.join(self.work_dir, AF2_PDB_name)
        # shared_AF2_PDB_path = os.path.join(shared_af2_pdb_dir, AF2_PDB_name)

        print('00000')
        if os.path.exists(AF2_PDB_path):

            self.get_af2_pic(AF2_PDB_path)

            # shutil.copy(AF2_PDB_path, shared_AF2_PDB_path)

            # 等待服务器回传图片
            AF2_png_name = AF2_PDB_name.replace(".pdb", ".png")
            AF2_png_path = os.path.join(self.work_dir, AF2_png_name)
            # shared_AF2_png_path = os.path.join(shared_af2_png_pir, AF2_png_name)
            # for times in range(30):
            #
            #     if os.path.exists(shared_AF2_png_path):
            #         shutil.copy(shared_AF2_png_path, AF2_png_path)
            #         break
            #
            #     time.sleep(10)

        else:
            AF2_png_path = None

        print(AF2_PDB_path)
        print(AF2_png_path)
        # print(shared_AF2_PDB_path)
        print('1111')
        # PPt 导入图片
        if AF2_png_path and os.path.exists(AF2_png_path):
            slice = ppt_file.slides.add_slide(ppt_file.slide_layouts[8])

            pptx_protein_title = slice.placeholders[18]
            title = pptx_protein_title.text_frame
            title_paragraph = title.add_paragraph()
            title_run = title.paragraphs[0].add_run()
            title_run.text = "AlphaFold2 Structure Prediction"
            title_run.font.size = Pt(28)
            title_run.font.bold = True
            title_run.font.color.rgb = RGBColor(146, 50, 34)
            title_run.font.name = "Times New Roman"

            # left = Cm(2.7)
            # top = Cm(3.12)
            # # height = Cm(15)
            # width = Cm(20)

            left = Cm(0)
            top = Cm(0)
            # height = Cm(15)
            width = Cm(33.77)

            pic = slice.shapes.add_picture(AF2_png_path, left, top, width=width)

            '''
            Pae2Domain
            '''
            print("Domain Predict", flush=True)
            # 下载PAE Json
            AF2_PAE_name = "AF2-{}.json".format(self.uniprot_id)
            # 判断是否下载过
            if not os.path.exists(os.path.join(self.work_dir, AF2_PAE_name)):
                for times in range(10):
                    try:
                        # 获取pdb
                        response = requests.get(
                            url="https://alphafold.ebi.ac.uk/files/AF-{}-F1-predicted_aligned_error_v6.json".format(
                                self.uniprot_id),
                            #proxies=proxies,
                            proxies=None,
                            headers={
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
                                "Referer": "https://alphafold.ebi.ac.uk/entry/{}".format(self.uniprot_id)
                            }
                        )

                        if response.status_code == 200:
                            with open(os.path.join(self.work_dir, AF2_PAE_name), "w", encoding="utf-8") as f:
                                f.write(response.text)

                            break
                    except:
                        time.sleep(3)

            AF2_PAE_path = os.path.join(self.work_dir, AF2_PAE_name)


            # 上传pdb文件和json文件
            domain_parser_dir = "AF2-{}".format(self.uniprot_id)
            shared_domain_parser_path = os.path.join(shared_domain_parser_input_dir, domain_parser_dir)
            shared_domain_pdb_path = os.path.join(shared_domain_parser_path, AF2_PDB_name)
            shared_domain_pae_path = os.path.join(shared_domain_parser_path, AF2_PAE_name)

            if not os.path.exists(shared_domain_parser_path):
                os.makedirs(shared_domain_parser_path)
            print(AF2_PDB_path, AF2_PAE_path)
            # 上传PDB文件, 上传pae json文件]
            if os.path.exists(AF2_PAE_path):
                shutil.copy(AF2_PAE_path, shared_domain_pae_path)
            if os.path.exists(AF2_PDB_path):
                print("上传PDB和PAE文件到服务器生成结构域图片", flush=True)
                shutil.copy(AF2_PDB_path, shared_domain_pdb_path)


                # 等待服务器回传图片
                domain_predict_name = "AF2-{}.finalDPAM.domains".format(self.uniprot_id)
                domain_predict_path = os.path.join(self.work_dir, domain_predict_name)
                shared_domain_predict_path = os.path.join(shared_domain_parser_output_dir, domain_predict_name)
                if os.path.exists(shared_domain_predict_path):
                    shutil.copy(shared_domain_predict_path, domain_predict_path)
                else:
                    for times in range(30):
                        # 配置接口地址（根据你的服务器调整 IP 或端口）
                        url = "http://10.51.14.16/upload/domain_predict"

                        # 你要上传的 .pdb 文件路径
                        pdb_file_name = os.path.basename(AF2_PDB_path)

                        # 使用 multipart/form-data 发送 POST 请求
                        with open(AF2_PDB_path, 'rb') as pdb_file:
                            files = {'file': (pdb_file_name, pdb_file, 'chemical/x-pdb')}
                            response = requests.post(url, files=files)

                        if response.status_code == 200:
                            result = response.json()
                            if result.get("success"):
                                domain_str = result.get("domain_str", "").strip()

                                # 处理 domain_str
                                # 先按逗号分割不同 domain
                                domains = [d.strip() for d in domain_str.split(",") if d.strip()]

                                with open(shared_domain_predict_path, "w") as f:
                                    for i, domain in enumerate(domains, start=1):

                                        # 如果一个 domain 内有多个片段
                                        if "_" in domain:
                                            segments = domain.split("_")
                                            formatted = ",".join(segments)
                                        else:
                                            formatted = domain

                                        line = f"D{i}\t{formatted}"

                                        # 只有不是最后一行才加换行符
                                        if i < len(domains):
                                            f.write(line + "\n")
                                        else:
                                            f.write(line)

                            else:
                                print("Domain prediction failed.")
                        else:
                            print("Request failed.")

                        print(shared_domain_predict_path)

                        if os.path.exists(shared_domain_predict_path):
                            shutil.copy(shared_domain_predict_path, domain_predict_path)
                            break
                        time.sleep(60)
            print('22222')
            # 更新结构域数据(domain_predict_path对应的文件可能会不存在, 导致报错)
            self.domain_list.append(["1-{}".format(len(self.protein_seq))])

            with open(domain_predict_path, "r", encoding="utf-8") as f_dpre:
                for line in f_dpre:
                    temp_domain_list = re.split("\s+", line.strip())[1].split(",")
                    self.domain_list.append(temp_domain_list)
            print('33333')
            af2_domain_str = "{} 可以设计的domain结构: ".format(self.gene_name, len(self.protein_seq))
            for domain in self.domain_list:
                # domain格式为列表
                if len(domain) > 1:
                    # af2_domain_str += ",".join(domain) + "(共表达)" + "; "
                    af2_domain_str += ",".join(domain) + "; "
                else:
                    af2_domain_str += ",".join(domain) + "; "

            alphafold2_prediction_text = slice.placeholders[10]

            alphafold2_prediction_text.text = "Based on the AlphaFold2, this institution has a ****** degree of credibility\n" + af2_domain_str

        else:
            slice = ppt_file.slides.add_slide(ppt_file.slide_layouts[8])
            pptx_protein_title = slice.placeholders[18]
            title = pptx_protein_title.text_frame
            title_paragraph = title.add_paragraph()
            title_run = title.paragraphs[0].add_run()
            title_run.text = "AlphaFold2 Structure Prediction"
            title_run.font.size = Pt(28)
            title_run.font.bold = True
            title_run.font.color.rgb = RGBColor(146, 50, 34)
            title_run.font.name = "Times New Roman"

    def ppt_alphafold2_PAE(self):
        ppt_file = self.ppt_file_handle

        AF2_PAE_name = "AF2-{}-PAE.json".format(self.uniprot_id)
        # 判断是否下载过
        if not os.path.exists(os.path.join(self.work_dir, AF2_PAE_name)):
            for times in range(10):
                try:
                    # 获取pdb
                    response = requests.get(
                        url="https://alphafold.ebi.ac.uk/files/AF-{}-F1-predicted_aligned_error_v6.json".format(
                            self.uniprot_id),
                        #proxies=proxies,
                        proxies=None,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
                            "Referer": "https://alphafold.ebi.ac.uk/entry/{}".format(self.uniprot_id)
                        }
                    )

                    if response.status_code == 200:
                        with open(os.path.join(self.work_dir, AF2_PAE_name), "w", encoding="utf-8") as f:
                            f.write(response.text)

                        break
                except:
                    time.sleep(3)

        # 生成PAE图片
        AF2_PAE_path = os.path.join(self.work_dir, AF2_PAE_name)
        AF2_PAE_png_name = AF2_PAE_name.replace(".json", ".png")
        AF2_PAE_png_path = os.path.join(self.work_dir, AF2_PAE_png_name)
        print(AF2_PAE_path,'11111')
        if os.path.exists(AF2_PAE_path):
            with open(AF2_PAE_path, "r", encoding="utf-8") as f:
                temp_dict = json.load(f)

            pae_value = temp_dict[0]["predicted_aligned_error"]

            # 画图
            sns.set()
            ax = sns.heatmap(pae_value, cmap="Greens_r")
            plt.xlabel('Scored residue', fontsize=10, color='k', family='Times New Roman')
            plt.ylabel('Aligned residue', fontsize=10, color='k', family='Times New Roman')
            cbar = ax.collections[0].colorbar
            cbar.set_label('Expected position error (Ångströms)', fontsize=10, family='Times New Roman')

            plt.savefig(AF2_PAE_png_path, dpi=1200, bbox_inches='tight')
            plt.clf()

        # PPt 导入图片
        if os.path.exists(AF2_PAE_png_path):
            slice = ppt_file.slides.add_slide(ppt_file.slide_layouts[17])
            pptx_protein_title = slice.placeholders[18]
            title = pptx_protein_title.text_frame
            title_paragraph = title.add_paragraph()
            title_run = title.paragraphs[0].add_run()
            title_run.text = "AlphaFold2 Predicted Aligned Error"
            title_run.font.size = Pt(28)
            title_run.font.bold = True
            title_run.font.color.rgb = RGBColor(146, 50, 34)
            title_run.font.name = "Times New Roman"

            # left = Cm(2.7)
            # top = Cm(3.12)
            # # height = Cm(15)
            # width = Cm(20)

            left = Cm(5.6)
            top = Cm(3.17)
            # height = Cm(15)
            width = Cm(14.41)

            pic = slice.shapes.add_picture(AF2_PAE_png_path, left, top, width=width)

        else:
            slice = ppt_file.slides.add_slide(ppt_file.slide_layouts[17])
            pptx_protein_title = slice.placeholders[18]
            title = pptx_protein_title.text_frame
            title_paragraph = title.add_paragraph()
            title_run = title.paragraphs[0].add_run()
            title_run.text = "AlphaFold2 Predicted Aligned Error"
            title_run.font.size = Pt(28)
            title_run.font.bold = True
            title_run.font.color.rgb = RGBColor(146, 50, 34)
            title_run.font.name = "Times New Roman"

    def ppt_protein_domain_analysis(self, defalut_download_path="", waittime=60):
        ppt_file = self.ppt_file_handle

        # 判断是否比对过
        print("判断是否比对过")
        print(os.path.exists(os.path.join(self.work_dir, r'blast2.png')))
        if not os.path.exists(os.path.join(self.work_dir, r'blast2.png')):
            # 删除已经存在的Alignment.xml
            for roots, dirs, files in os.walk(self.work_dir):
                for file in files:
                    if re.search("-Alignment\.xml$", file):
                        os.remove(os.path.join(roots, file))

            try:

                # 初始化浏览器
                bro = webdriver.Chrome(executable_path=self.chrome_driver_path)

                # 设置浏览器默认下载和阻止危害文件提醒弹出窗口
                options = webdriver.ChromeOptions()
                prefs = {'download.prompt_for_download': False, 'download.default_directory': defalut_download_path}
                options.add_experimental_option("prefs", prefs)
                bro = webdriver.Chrome(chrome_options=options, executable_path=self.chrome_driver_path)

                bro.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')
                params = {'cmd': 'Page.setDownloadBehavior',
                          'params': {'behavior': 'allow', 'downloadPath': defalut_download_path}}
                bro.execute("send_command", params)

                # 让浏览器对指定url发起访问,访问NCBI蛋白质比对网址
                bro.get(
                    "https://blast.ncbi.nlm.nih.gov/Blast.cgi?PROGRAM=blastp&PAGE_TYPE=BlastSearch&LINK_LOC=blasthome")

                bro.maximize_window()

                time.sleep(10)

                # 等待输入框加载，输入NCBI号
                for times in range(10):
                    try:
                        WebDriverWait(bro, waittime).until(
                            EC.presence_of_element_located((By.XPATH, '//*[@id="seq"]'))
                        ).send_keys(self.protein_seq + "\n")

                        break
                    except:

                        time.sleep(3)

                # 选择pdb数据库
                for times in range(10):
                    try:
                        pdb_db_select = Select(bro.find_element_by_xpath('//*[@id="DATABASE"]'))
                        pdb_db_select.select_by_value("pdb")

                        break
                    except:

                         time.sleep(3)

                # 选择比对算法
                for times in range(10):
                    try:
                        WebDriverWait(bro, waittime).until(
                            EC.element_to_be_clickable(
                                (By.XPATH, '//*[@id="progSel"]/div[2]/table/tbody/tr[2]/td/label'))
                        ).click()
                        break
                    except:

                        time.sleep(3)

                # 提交比对
                for times in range(10):
                    try:
                        WebDriverWait(bro, waittime).until(
                            EC.element_to_be_clickable(
                                (By.XPATH, '//*[@id="blastButton1"]/input'))
                        ).click()
                        break
                    except:

                        time.sleep(3)

                # 等待比对完成(下载按钮出现)
                for times in range(10):
                    try:
                        WebDriverWait(bro, waittime).until(
                            EC.element_to_be_clickable(
                                (By.XPATH, '//*[@id="ulDnldAl"]'))
                        ).click()
                        break
                    except:

                        time.sleep(3)

                # 下载XML
                for times in range(10):
                    try:
                        WebDriverWait(bro, waittime).until(
                            EC.element_to_be_clickable(
                                (By.XPATH, '//*[@id="allDownload"]/li[2]/a'))
                        ).click()
                        break
                    except:

                        time.sleep(3)

                # 检查XML是否下载完毕
                xml_download_status = False

                # while True:
                for i in range(30):
                    time.sleep(10)

                    for roots, dirs, files in os.walk(self.work_dir):
                        for file in files:
                            if re.search("-Alignment\.xml$", file):
                                xml_download_status = True

                    if xml_download_status:
                        break

                # 截图比对结果
                blast_ele1 = bro.find_element_by_xpath('//*[@id="dscSort"]')

                js4 = "arguments[0].scrollIntoView();"

                bro.execute_script(js4, blast_ele1)

                time.sleep(5)

                bro.get_screenshot_as_file(os.path.join(self.work_dir, r'blast1.png'))

                time.sleep(3)

                WebDriverWait(bro, waittime).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, '//*[@id="btnGrph"]'))
                ).click()

                blast_ele2 = bro.find_element_by_xpath('//*[@id="grBlastHits"]/h3')

                js4 = "arguments[0].scrollIntoView();"

                bro.execute_script(js4, blast_ele2)

                time.sleep(5)

                bro.get_screenshot_as_file(os.path.join(self.work_dir, r'blast2.png'))  # 路径不存在不报错，但保存不了

                bro.quit()

                # Modify pic
                im = Image.open(os.path.join(self.work_dir, r'blast1.png'))
                im = im.crop((0, 0, im.size[0], 450))
                im.save(os.path.join(self.work_dir, r'blast1_modified.png'))

                im = Image.open(os.path.join(self.work_dir, r'blast2.png'))
                im = im.crop((0, 0, im.size[0], 550))
                im.save(os.path.join(self.work_dir, r'blast2_modified.png'))

                # add ppt slice
                slice = ppt_file.slides.add_slide(ppt_file.slide_layouts[9])
                pptx_protein_title = slice.placeholders[18]
                title = pptx_protein_title.text_frame
                title_paragraph = title.add_paragraph()
                title_run = title.paragraphs[0].add_run()
                title_run.text = "Protein Domain Analysis"
                title_run.font.size = Pt(28)
                title_run.font.bold = True
                title_run.font.color.rgb = RGBColor(146, 50, 34)
                title_run.font.name = "Times New Roman"

                left = Cm(0.18)
                top = Cm(10.25)
                # height = Cm(15)
                width = Cm(25)
                pic = slice.shapes.add_picture(os.path.join(defalut_download_path, r'blast1_modified.png'), left, top,
                                               width=width)

                left = Cm(0.18)
                top = Cm(2.85)
                # height = Cm(15)
                width = Cm(25)
                pic = slice.shapes.add_picture(os.path.join(defalut_download_path, r'blast2_modified.png'), left, top,
                                               width=width)

                alphafold2_prediction_text = slice.placeholders[10]

                alphafold2_prediction_text.text = "This protein has ****** structures in PDB with above ****** identities for its sequence. \nIt shouble be a ****** reference for this project."

            except:
                pass

        else:

            # Modify pic
            im = Image.open(os.path.join(self.work_dir, r'blast1.png'))
            im = im.crop((0, 0, im.size[0], 450))
            im.save(os.path.join(self.work_dir, r'blast1_modified.png'))

            im = Image.open(os.path.join(self.work_dir, r'blast2.png'))
            im = im.crop((0, 0, im.size[0], 550))
            im.save(os.path.join(self.work_dir, r'blast2_modified.png'))

            # add ppt slice
            slice = ppt_file.slides.add_slide(ppt_file.slide_layouts[9])
            pptx_protein_title = slice.placeholders[18]
            title = pptx_protein_title.text_frame
            title_paragraph = title.add_paragraph()
            title_run = title.paragraphs[0].add_run()
            title_run.text = "Protein Domain Analysis"
            title_run.font.size = Pt(28)
            title_run.font.bold = True
            title_run.font.color.rgb = RGBColor(146, 50, 34)
            title_run.font.name = "Times New Roman"

            left = Cm(0.18)
            top = Cm(10.25)
            # height = Cm(15)
            width = Cm(25)
            pic = slice.shapes.add_picture(os.path.join(defalut_download_path, r'blast1_modified.png'), left, top,
                                           width=width)

            left = Cm(0.18)
            top = Cm(2.85)
            # height = Cm(15)
            width = Cm(25)
            pic = slice.shapes.add_picture(os.path.join(defalut_download_path, r'blast2_modified.png'), left, top,
                                           width=width)

            alphafold2_prediction_text = slice.placeholders[10]

            alphafold2_prediction_text.text = "This protein has ****** structures in PDB with above ****** identities for its sequence. \nIt shouble be a ****** reference for this project."

    def ppt_uniprot_pdb_summary(self, columns=6):
        ppt_file = self.ppt_file_handle

        # 字典, 记录uniprot_pdb
        uniprot_pdb_info_list = []

        # # get pdb id
        # for roots,dirs,files in os.walk(work_dir):
        #     for file in files:
        #         if re.search("\.xml$",file):
        #             blast_xml_file = os.path.join(roots,file)
        #
        # with open(blast_xml_file,"r",encoding="utf-8") as f:
        #     blast_xml_content = f.read()
        uniprot_pdb_list = self.uniprot_content_dict["pdb_summary_list"]

        # rows = len(uniprot_pdb_list) + 1
        rows = 11  # 最大写入10行

        slice = ppt_file.slides.add_slide(ppt_file.slide_layouts[16])
        pptx_protein_title = slice.placeholders[18]
        title = pptx_protein_title.text_frame
        title_paragraph = title.add_paragraph()
        title_run = title.paragraphs[0].add_run()
        title_run.text = "Uniprot PDB Summary"
        title_run.font.size = Pt(28)
        title_run.font.bold = True
        title_run.font.color.rgb = RGBColor(146, 50, 34)
        title_run.font.name = "Times New Roman"

        ppt_pdb_summary_table = slice.placeholders[12]

        # protein table
        table = ppt_pdb_summary_table.insert_table(rows, columns).table

        # row height
        for row in table.rows:
            row.height = 150000
            # print(row.height)

        ## protein table header
        table.cell(0,0).text,table.cell(0,1).text,table.cell(0,2).text,table.cell(0,3).text,table.cell(0,4).text,table.cell(0,5).text = \
            "PDB","Space Group","Resolution","Unit Cell","Ligands","Stoichiometry"
        # table.cell(0, 0).text_frame.paragraphs.font.size = Pt(14)

        # # 解析pdb id, info
        # soup = BeautifulSoup(blast_xml_content,"xml")
        #
        # Hits_list = soup.find_all(name="Hit")

        PDB_write_status = 1
        for Hit_index, Hit in enumerate(uniprot_pdb_list):

            Hit_id = Hit
            print("获取{0}解析方法, pdb url:".format(Hit_id), "https://data.rcsb.org/rest/v1/core/entry/{0}".format(Hit_id), flush=True)

            if os.path.exists(os.path.join(self.work_dir, "crystal_info_{0}.json".format(Hit_id))):
                with open(os.path.join(self.work_dir, "crystal_info_{0}.json".format(Hit_id)), "r", encoding="utf-8") as f:
                    crystal_info_dict = json.load(f)

            else:
                for times in range(10):
                    try:
                        print("  -->第{}尝试".format(times), flush=True)
                        crystal_response = requests.get(
                            url="https://data.rcsb.org/rest/v1/core/entry/{0}".format(Hit_id),
                            proxies={},
                            headers={
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
                                "Referer": "https://www.rcsb.org/",

                            }
                        )



                        if crystal_response.status_code == 200:
                            break

                    except Exception as e:
                        print(e)
                        print('获取失败*******')
                        time.sleep(5)

                else:
                    # 网络连接失败, 继续下一个
                    continue

                with open(os.path.join(self.work_dir, "crystal_info_{0}.json".format(Hit_id)), "w", encoding="utf-8") as f:
                    f.write(crystal_response.text)

                with open(os.path.join(self.work_dir, "crystal_info_{0}.json".format(Hit_id)), "r", encoding="utf-8") as f:
                    crystal_info_dict = json.load(f)

            # 判断PDB是否为Xray（晶体）或ELECTRON MICROSCOPY（电镜）解析出的结构
            structure_method = ""
            try:
                # 新版 RCSB JSON（推荐）
                if crystal_info_dict["rcsb_entry_info"]["experimental_method"]:
                    method = crystal_info_dict["rcsb_entry_info"]["experimental_method"].upper().strip()

                    if "X-RAY" in method:
                        structure_method = "Xray"
                    elif "X-ray" in method:
                        structure_method = "Xray"
                    elif "ELECTRON" in method:
                        structure_method = "ELECTRON MICROSCOPY"
                    
                    elif "EM" in method:
                        structure_method = "ELECTRON MICROSCOPY"
                    
                    else:
                        structure_method = method

            except Exception as e:
                try:
                    # 兼容老版 JSON
                    if crystal_info_dict["symmetry"]["space_group_name_H_M"]:
                        structure_method = "Xray"
                except Exception as e:
                    try:
                        # 更老版本
                        if crystal_info_dict["symmetry"]["space_group_name_hm"]:
                            structure_method = "Xray"
                    except Exception as e:
                        try:
                            # 最后再使用 exptl
                            method = crystal_info_dict["exptl"][0]["method"].upper().strip()

                            if "X-RAY" in method:
                                structure_method = "Xray"
                            elif "X-ray" in method:
                                structure_method = "Xray"
                            elif "ELECTRON" in method:
                                structure_method = "ELECTRON MICROSCOPY"
                            elif "EM" in method:
                                structure_method = "ELECTRON MICROSCOPY"
                            else:
                                structure_method = method

                        except Exception as e:
                            print(e)
                            structure_method = "structure method not find"

            print("  -->解析方法为: ", structure_method, flush=True)
            print("获取{0}解析参数, pdb url:".format(Hit_id), "https://www.rcsb.org/structure/{0}".format(Hit_id), flush=True)

            if structure_method == "Xray":
                print("  -->解析方法为Xray, 获取晶体参数", flush=True)
                for times in range(10):
                    try:
                        print('11111',os.path.exists(os.path.join(self.work_dir, r'{}.html'.format(Hit_id))), flush=True)
                        if not os.path.exists(os.path.join(self.work_dir, r'{}.html'.format(Hit_id))):
                            print("  -->第{}尝试".format(times), flush=True)
                            Stoichiometry_response = requests.get(
                                url="https://www.rcsb.org/structure/{0}".format(Hit_id),
                                proxies={},
                                headers={
                                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
                                    "Referer": "https://www.rcsb.org/",

                                }
                            )

                            if Stoichiometry_response.status_code == 200:
                                with open(os.path.join(self.work_dir, r'{}.html'.format(Hit_id)), "w", encoding="utf-8") as f_html:
                                    f_html.write(Stoichiometry_response.text)

                                Stoichiometry_soup = BeautifulSoup(Stoichiometry_response.text, "html.parser")

                               

                                Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"carousel-footer"})
                                if Stoichiometry_text is None:
                                    Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"card-footer"})
                           
                                Stoichiometry_info = Stoichiometry_text.text.split("Find")[0].split("Global Stoichiometry:")[1].strip()

                            else:
                                # 网络失败
                                print('*******')
                                Stoichiometry_info = ""

                        else:
                            with open(os.path.join(self.work_dir, r'{}.html'.format(Hit_id)), "r", encoding="utf-8") as f_html:
                                hitid_html = f_html.read()

                            Stoichiometry_soup = BeautifulSoup(hitid_html, "html.parser")

                            Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"carousel-footer"})
                            if Stoichiometry_text is None:
                                Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"card-footer"})
                            Stoichiometry_info = Stoichiometry_text.text.split("Find")[0].split("Global Stoichiometry:")[1].strip()

                        try:
                            ligand_name_list = []
                            if crystal_info_dict["rcsb_entry_container_identifiers"]["non_polymer_entity_ids"]:
                                non_polymer_entity_ids = crystal_info_dict["rcsb_entry_container_identifiers"]["non_polymer_entity_ids"]

                                # ligand_name_list = crystal_info_dict["pdbx_vrpt_summary"]["restypes_notchecked_for_bond_angle_geometry"]
                                for non_polymer_entity_id in non_polymer_entity_ids:
                                    ligand_name = self.get_pdb_ligand_name(Hit_id, non_polymer_entity_id)
                                    if ligand_name:
                                        ligand_name_list.append(ligand_name)
                        except:
                            ligand_name_list = []
                        # print(Hit_id,Truncation_length_begin + "-" + Truncation_length_end + " aa" + "({0})".format(identities))
                        try:
                            # 保存所有的Pdb信息(写入到Excel表格中)
                            try:                                 
                                    try:
                                        space_group = crystal_info_dict["symmetry"]["space_group_name_hm"]
                                    except:
                                        try:
                                            space_group = crystal_info_dict["symmetry"]["space_group_name_H_M"]
                                        except:
                                            space_group = ""
                                            print("space_group 没找到，检查是否改名字或者不存在", flush=True)

                                    try:
                                        resolution = str(
                                            crystal_info_dict["rcsb_entry_info"]["resolution_combined"][0]
                                        ) + " Å"
                                    except:
                                        resolution = ""
                                        print("resolution 没找到，检查是否改名字或者不存在", flush=True)

                                    try:
                                        unit_cell = (
                                            str(crystal_info_dict["cell"]["length_a"]) + ", " +
                                            str(crystal_info_dict["cell"]["length_b"]) + ", " +
                                            str(crystal_info_dict["cell"]["length_c"])
                                        )
                                    except:
                                        unit_cell = ""
                                        print("unit_cell 没找到，检查是否改名字或者不存在", flush=True)

                                    uniprot_pdb_info_list.append({
                                        "Hit_id": Hit_id,
                                        "Space Group": space_group,
                                        "Resolution": resolution,
                                        "Unit Cell": unit_cell,
                                        "Ligands": ", ".join(ligand_name_list),
                                        "Stoichiometry": Stoichiometry_info
                                    })

                            except Exception as e:
                                    print("保存PDB信息失败：{}".format(e), flush=True)

                            # try:
                            #     uniprot_pdb_info_list.append({"Hit_id": Hit_id,
                            #                          "Space Group": crystal_info_dict["symmetry"]["space_group_name_hm"],
                            #                          "Resolution": str(crystal_info_dict["rcsb_entry_info"]["resolution_combined"][0]) + " Å",
                            #                          "Unit Cell": str(crystal_info_dict["cell"]['length_a']) + ", " + str(crystal_info_dict["cell"]['length_b']) + ", " + str(crystal_info_dict["cell"]['length_c']),
                            #                          "Ligands": ", ".join(ligand_name_list),
                            #                          "Stoichiometry": Stoichiometry_info})
                            # except:
                            #      uniprot_pdb_info_list.append({"Hit_id": Hit_id,
                            #                                     "Space Group": crystal_info_dict["symmetry"]["space_group_name_H_M"],
                            #                                     "Resolution": str(crystal_info_dict["rcsb_entry_info"]["resolution_combined"][0]) + " Å",
                            #                                     "Unit Cell": str(crystal_info_dict["cell"]['length_a']) + ", " + str(crystal_info_dict["cell"]['length_b']) + ", " + str(crystal_info_dict["cell"]['length_c']),
                            #                                     "Ligands": ", ".join(ligand_name_list),
                            #                                     "Stoichiometry": Stoichiometry_info})

                            #print("  -->获取{0}解析参数成功, pdb url:".format(Hit_id), "https://www.rcsb.org/structure/{0}".format(Hit_id), flush=True)
                            # 只记录10条
                            if PDB_write_status >= 11:
                                pass
                            else:
                                try:   
                                        try:
                                            space_group = crystal_info_dict["symmetry"]["space_group_name_hm"]
                                        except:
                                            try:
                                                space_group = crystal_info_dict["symmetry"]["space_group_name_H_M"]
                                            except:
                                                space_group = ""

                                        try:
                                            resolution = str(
                                                crystal_info_dict["rcsb_entry_info"]["resolution_combined"][0]
                                            ) + " Å"
                                        except:
                                            resolution = ""

                                        try:
                                            unit_cell = (
                                                str(crystal_info_dict["cell"]["length_a"]) + ", " +
                                                str(crystal_info_dict["cell"]["length_b"]) + ", " +
                                                str(crystal_info_dict["cell"]["length_c"])
                                            )
                                        except:
                                            unit_cell = ""

                                        uniprot_pdb_info_list.append({
                                            "Hit_id": Hit_id,
                                            "Space Group": space_group,
                                            "Resolution": resolution,
                                            "Unit Cell": unit_cell,
                                            "Ligands": ", ".join(ligand_name_list),
                                            "Stoichiometry": Stoichiometry_info
                                        })

                                except Exception as e:
                                    print("保存PDB信息失败：{}".format(e), flush=True)
                                # try:
                                #     table.cell(PDB_write_status, 0).text, table.cell(PDB_write_status, 1).text, table.cell(PDB_write_status, 2).text, table.cell(PDB_write_status, 3).text, table.cell(PDB_write_status,
                                #                                                                                                        4).text, table.cell(
                                #     PDB_write_status, 5).text = \
                                #     Hit_id, crystal_info_dict["symmetry"]["space_group_name_hm"], str(crystal_info_dict["rcsb_entry_info"]["resolution_combined"][0]) + " Å", \
                                #     str(crystal_info_dict["cell"]['length_a']) + ", " + str(crystal_info_dict["cell"]['length_b']) + ", " + str(crystal_info_dict["cell"]['length_c']), ", ".join(ligand_name_list), Stoichiometry_info
                                # except:
                                #     table.cell(PDB_write_status, 0).text, table.cell(PDB_write_status, 1).text, table.cell(PDB_write_status, 2).text, table.cell(PDB_write_status, 3).text, table.cell(PDB_write_status,
                                #                                                                                                                                            4).text, table.cell(
                                #                                         PDB_write_status, 5).text = \
                                #                                         Hit_id, crystal_info_dict["symmetry"]["space_group_name_H_M"], str(crystal_info_dict["rcsb_entry_info"]["resolution_combined"][0]) + " Å", \
                                #                                     str(crystal_info_dict["cell"]['length_a']) + ", " + str(crystal_info_dict["cell"]['length_b']) + ", " + str(crystal_info_dict["cell"]['length_c']), ", ".join(ligand_name_list), Stoichiometry_info
                                    
                        except Exception as e:
                            print("错误信息", e, flush=True)
                            print(Hit_id, flush=True)
                            # input("wait:")

                        break

                    except Exception as e:
                        print("错误信息", e, flush=True)
                        pass

                # PDB 信息写入的记录加1
                PDB_write_status += 1

            elif structure_method == "ELECTRON MICROSCOPY":
                print("  -->解析方法为ELECTRON MICROSCOPY, 获取晶体参数", flush=True)
                for times in range(10):
                    try:
                        print('11111',os.path.exists(os.path.join(self.work_dir, r'{}.html'.format(Hit_id))), flush=True)
                        if not os.path.exists(os.path.join(self.work_dir, r'{}.html'.format(Hit_id))):
                            print("  -->第{}尝试".format(times), flush=True)
                            Stoichiometry_response = requests.get(
                                url="https://www.rcsb.org/structure/{0}".format(Hit_id),
                                #proxies=proxies,
                                proxies=None,
                                headers={
                                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
                                    "Referer": "https://www.rcsb.org/",

                                }
                            )
                            if Stoichiometry_response.status_code == 200:
                                with open(os.path.join(self.work_dir, r'{}.html'.format(Hit_id)), "w",
                                          encoding="utf-8") as f_html:
                                    f_html.write(Stoichiometry_response.text)

                                Stoichiometry_soup = BeautifulSoup(Stoichiometry_response.text, "html.parser")

                                Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"carousel-footer"})
                                if Stoichiometry_text is None:
                                    Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"card-footer"})

                                Stoichiometry_info = \
                                    Stoichiometry_text.text.split("Find")[0].split("Global Stoichiometry:")[1].strip()

                            else:
                                # 网络失败
                                Stoichiometry_info = ""

                        else:
                            with open(os.path.join(self.work_dir, r'{}.html'.format(Hit_id)), "r", encoding="utf-8") as f_html:
                                hitid_html = f_html.read()
                            
                            Stoichiometry_soup = BeautifulSoup(hitid_html, "html.parser")
                            # print(111111, flush=True)
                            # print(type(Stoichiometry_soup), flush=True)
                            
                            # Stoichiometry_div = Stoichiometry_soup.find(
                            #     name="div",
                            #     attrs={"id": "Carousel-BiologicalUnit1"}
                            # )

                            # if Stoichiometry_div is not None:
                            #     print("找到 Carousel-BiologicalUnit1", flush=True)
                            # else:
                            #     print("没有找到 Carousel-BiologicalUnit1", flush=True)
                                
                            Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"carousel-footer"})
                            print("111111", flush=True)
                            if Stoichiometry_text is None:
                                Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"card-footer"})
                                print("Stoichiometry_text is None, check the HTML structure for Hit_id: {}".format(Hit_id), flush=True)
                                print("2222222", flush=True)
                            Stoichiometry_info = Stoichiometry_text.text.split("Find")[0].split("Global Stoichiometry:")[1].strip()


                        try:
                            ligand_name_list = []
                            if crystal_info_dict["rcsb_entry_container_identifiers"]["non_polymer_entity_ids"]:
                                non_polymer_entity_ids = crystal_info_dict["rcsb_entry_container_identifiers"][
                                    "non_polymer_entity_ids"]

                                # ligand_name_list = crystal_info_dict["pdbx_vrpt_summary"]["restypes_notchecked_for_bond_angle_geometry"]
                                for non_polymer_entity_id in non_polymer_entity_ids:
                                    ligand_name = self.get_pdb_ligand_name(Hit_id, non_polymer_entity_id)
                                    if ligand_name:
                                        ligand_name_list.append(ligand_name)
                        except:
                            ligand_name_list = []
                        uniprot_pdb_info_list.append({"Hit_id": Hit_id,
                                                      "Space Group": "",
                                                      "Resolution": str(
                                                          crystal_info_dict["rcsb_entry_info"]["resolution_combined"][0]) + " Å",
                                                      "Unit Cell": "",
                                                      "Ligands": ", ".join(ligand_name_list),
                                                      "Stoichiometry": Stoichiometry_info})
                        # 只记录10条
                        if PDB_write_status >= 11:
                            pass
                        else:
                            table.cell(PDB_write_status, 0).text, table.cell(PDB_write_status, 1).text, table.cell(
                                PDB_write_status, 2).text, table.cell(PDB_write_status, 3).text, table.cell(PDB_write_status,
                                                                                                            4).text, table.cell(
                                PDB_write_status, 5).text = \
                                Hit_id, "", str(
                                    crystal_info_dict["rcsb_entry_info"]["resolution_combined"][0]) + " Å", \
                                "", ", ".join(ligand_name_list), Stoichiometry_info

                        break

                    except Exception as e:
                        print("错误信息", e, flush=True)
                        pass

                # PDB 信息写入的记录加1
                PDB_write_status += 1

        ## protein table format
        for cell in table.iter_cells():
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(12)
                # paragraph.line_spacing = 1
                paragraph.font.name = "Times New Roman"
                paragraph.alignment = PP_ALIGN.CENTER
                paragraph.font.color.rgb = RGBColor(0, 0, 0)

        # 保存PPT
        pptx_save_name = "{0}_{1}_{2}_Biortus_Structure_Evaluation Report_{3}.pptx".format(self.uniprot_id,
                         time.strftime('%Y%m%d', time.localtime()), self.gene_name, self.person_name).replace('/','_')
        pptx_save_path = os.path.join(r'D:\1项目评估自动化\ppt', pptx_save_name)
        ppt_file.save(pptx_save_path)

        "插入表格对象"
        # 写表格
        wb = openpyxl.Workbook()
        ws = wb.worksheets[0]

        # 写标题
        title_list = ["PDB", "Space Group", "Resolution", "Unit Cell", "Ligands", "Stoichiometry"]
        title_row = 1
        title_column = 1
        for title in title_list:
            ws.cell(title_row, title_column, value=title)
            # 字体
            ws[get_column_letter(title_column) + str(title_row)].font = title_font
            # 对齐
            ws[get_column_letter(title_column) + str(title_row)].alignment = content_align
            # 列宽
            # worksheet.column_dimensions[get_column_letter(column_num)].width = sys.getsizeof(value) * column_unit_length
            ws.column_dimensions[get_column_letter(title_column)].width = 55.6

            title_column += 1

        # 写内容
        data_row = 2
        for uniprot_pdb_info_dict in uniprot_pdb_info_list:
            data_column = 1
            for uniprot_pdb_info in uniprot_pdb_info_dict.values():
                ws.cell(data_row, data_column, value=uniprot_pdb_info)
                # 字体
                ws[get_column_letter(data_column) + str(data_row)].font = content_font
                # 对齐
                ws[get_column_letter(data_column) + str(data_row)].alignment = content_align

                data_column += 1

            data_row += 1

        wb.save(os.path.join(self.work_dir, "uniprot_pdb_summary.xlsx"))

        for excel_i in range(10):
            # 插入表格对象
            # 导入PPT 模块
            ppt_app = Dispatch('PowerPoint.Application')
            ppt_template = ppt_app.Presentations.Open(pptx_save_path, WithWindow=False)
            try:
                print("第{}次插入表格".format(excel_i), flush=True)
                # print(ppt_template.Slides.Count)
                ppt_template.Slides(ppt_template.Slides.Count).Shapes.AddOLEObject(625, 25, 70, 70,
                                                                           FileName=os.path.join(self.work_dir, "uniprot_pdb_summary.xlsx"),
                                                                           IconLabel="PdbSummary", DisplayAsIcon=True)

                break

            except Exception as e:
                print("插入表格失败, 错误信息:{}".format(e), flush=True)
                pass
                time.sleep(3)

        # 保存目标PPT
        ppt_template.SaveAs(pptx_save_path)
        print('000000')
        ppt_template.Close()

        print('111111')
        print('111111')


        # 重新复制ppt操作对象
        self.ppt_file_handle = Presentation(pptx_save_path)

    def ppt_pdb_summary(self, rows=11,columns=7):
        ppt_file = self.ppt_file_handle

        # get pdb id
        for roots,dirs,files in os.walk(self.work_dir):
            for file in files:
                if re.search("-Alignment\.xml$", file):
                    # print("ppt_pdb_summary", file)
                    blast_xml_file = os.path.join(roots, file)

        with open(blast_xml_file, "r", encoding="utf-8") as f:
            blast_xml_content = f.read()


        slice = ppt_file.slides.add_slide(ppt_file.slide_layouts[10])
        pptx_protein_title = slice.placeholders[18]
        title = pptx_protein_title.text_frame
        title_paragraph = title.add_paragraph()
        title_run = title.paragraphs[0].add_run()
        title_run.text = "PDB Summary"
        title_run.font.size = Pt(28)
        title_run.font.bold = True
        title_run.font.color.rgb = RGBColor(146, 50, 34)
        title_run.font.name = "Times New Roman"


        ppt_pdb_summary_table = slice.placeholders[12]

        # protein table
        table = ppt_pdb_summary_table.insert_table(rows, columns).table

        # row height
        for row in table.rows:
            row.height = 150000
            # print(row.height)

        ## protein table header
        table.cell(0,0).text,table.cell(0,1).text,table.cell(0,2).text,table.cell(0,3).text,table.cell(0,4).text,table.cell(0,5).text,table.cell(0,6).text = \
            "PDB","Space Group","Resolution","Unit Cell","Truncation(identities)","Ligands","Stoichiometry"
        # table.cell(0, 0).text_frame.paragraphs.font.size = Pt(14)

        # 解析pdb id, info
        soup = BeautifulSoup(blast_xml_content,"xml")

        Hits_list = soup.find_all(name="Hit")

        PDB_write_status = 1
        for Hit_index, Hit in enumerate(Hits_list):
            # get top 5 PDB
            if PDB_write_status >= 11:
                break
            print("  -->{}".format(Hit), flush=True)
            Hit_id = Hit.find(name="Hit_id").text.split("|")[1]
            Truncation_length_begin = Hit.find(name="Hsp_query-from").text
            Truncation_length_end = Hit.find(name="Hsp_query-to").text
            Hit_identity = Hit.find(name="Hsp_identity").text
            Hit_align_length = Hit.find(name="Hsp_align-len").text
            identities = Truncation_length_begin + "-" + Truncation_length_end + " aa(" + str(round(int(Hit_identity) / int(Hit_align_length) * 100,2)) + "%)"
            # get crystal_info
            print("获取{0}解析方法, pdb url:".format(Hit_id), "https://data.rcsb.org/rest/v1/core/entry/{0}".format(Hit_id), flush=True)

            if os.path.exists(os.path.join(self.work_dir, "crystal_info_{0}.json".format(Hit_id))):
                with open(os.path.join(self.work_dir, "crystal_info_{0}.json".format(Hit_id)), "r", encoding="utf-8") as f:
                    crystal_info_dict = json.load(f)

            else:
                for times in range(10):
                    try:
                        print("  -->第{}尝试".format(times), flush=True)
                        crystal_response = requests.get(
                            url="https://data.rcsb.org/rest/v1/core/entry/{0}".format(Hit_id),
                            proxies={},
                            headers={
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
                                "Referer": "https://www.rcsb.org/",

                            }
                        )
                        if crystal_response.status_code == 200:
                            break

                    except:
                        time.sleep(5)
                else:
                    # 网络连接失败, 继续下一个
                    continue

                with open(os.path.join(self.work_dir, "crystal_info_{0}.json".format(Hit_id)), "w", encoding="utf-8") as f:
                    f.write(crystal_response.text)

                with open(os.path.join(self.work_dir, "crystal_info_{0}.json".format(Hit_id)), "r", encoding="utf-8") as f:
                    crystal_info_dict = json.load(f)

            # 判断PDB是否为Xray或其他实验方法解析出的结构
            structure_method = ""
            try:
                # 新版 RCSB JSON（推荐）
                if crystal_info_dict["rcsb_entry_info"]["experimental_method"]:
                    method = crystal_info_dict["rcsb_entry_info"]["experimental_method"].upper().strip()

                    if "X-RAY" in method:
                        structure_method = "Xray"
                    elif "X-ray" in method:
                        structure_method = "Xray"
                    elif "ELECTRON" in method:
                        structure_method = "ELECTRON MICROSCOPY"
                    elif "EM" in method:
                        structure_method = "ELECTRON MICROSCOPY"#experimental_method":"EM",冷冻电镜
                    else:
                        structure_method = method

            except Exception as e:
                try:
                    # 兼容老版 JSON
                    if crystal_info_dict["symmetry"]["space_group_name_H_M"]:
                        structure_method = "Xray"
                except Exception as e:
                    try:
                        # 更老版本
                        if crystal_info_dict["symmetry"]["space_group_name_hm"]:
                            structure_method = "Xray"
                    except Exception as e:
                        try:
                            # 最后再使用 exptl
                            method = crystal_info_dict["exptl"][0]["method"].upper().strip()

                            if "X-RAY" in method:
                                structure_method = "Xray"
                            elif "X-ray" in method:
                                structure_method = "Xray"
                            elif "ELECTRON" in method:
                                structure_method = "ELECTRON MICROSCOPY"
                            elif "EM" in method:
                                structure_method = "ELECTRON MICROSCOPY"
                            
                            else:
                                structure_method = method

                        except Exception as e:
                            print(e)
                            structure_method = "structure method not find"

            if structure_method == "Xray":
                for times in range(10):
                    try:
                        if not os.path.exists(os.path.join(self.work_dir, r'{}.html'.format(Hit_id))):
                            print("  -->第{}尝试".format(times), flush=True)

                            Stoichiometry_response = requests.get(
                                url="https://www.rcsb.org/structure/{0}".format(Hit_id),
                                #proxies=proxies,
                                proxies=None,
                                headers={
                                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
                                    "Referer": "https://www.rcsb.org/",

                                }
                            )
                            if Stoichiometry_response.status_code == 200:
                                with open(os.path.join(self.work_dir, r'{}.html'.format(Hit_id)), "w", encoding="utf-8") as f_html:
                                    f_html.write(Stoichiometry_response.text)

                                Stoichiometry_soup = BeautifulSoup(Stoichiometry_response.text, "html.parser")

                                Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"carousel-footer"})
                                if Stoichiometry_text is None:
                                    Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"card-footer"})

                                Stoichiometry_info = \
                                Stoichiometry_text.text.split("Find")[0].split("Global Stoichiometry:")[1].strip()

                            else:
                                # 网络失败
                                print('*********')
                                Stoichiometry_info = ""
                        else:
                            with open(os.path.join(self.work_dir, r'{}.html'.format(Hit_id)), "r", encoding="utf-8") as f_html:
                                hitid_html = f_html.read()

                            Stoichiometry_soup = BeautifulSoup(hitid_html, "html.parser")

                            Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"carousel-footer"})
                            if Stoichiometry_text is None:
                                Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"card-footer"})
                            Stoichiometry_info = Stoichiometry_text.text.split("Find")[0].split("Global Stoichiometry:")[1].strip()

                        try:
                            ligand_name_list = []
                            if crystal_info_dict["rcsb_entry_container_identifiers"]["non_polymer_entity_ids"]:
                                non_polymer_entity_ids = crystal_info_dict["rcsb_entry_container_identifiers"]["non_polymer_entity_ids"]

                                # ligand_name_list = crystal_info_dict["pdbx_vrpt_summary"]["restypes_notchecked_for_bond_angle_geometry"]
                                for non_polymer_entity_id in non_polymer_entity_ids:
                                    ligand_name = self.get_pdb_ligand_name(Hit_id, non_polymer_entity_id)
                                    if ligand_name:
                                        ligand_name_list.append(ligand_name)
                        except:
                            ligand_name_list = []

                        # print(Hit_id,Truncation_length_begin + "-" + Truncation_length_end + " aa" + "({0})".format(identities))
                        try:
                            table.cell(PDB_write_status, 0).text, table.cell(PDB_write_status, 1).text, table.cell(PDB_write_status, 2).text, table.cell(PDB_write_status, 3).text, table.cell(PDB_write_status,4).text, table.cell(PDB_write_status, 5).text, table.cell(PDB_write_status, 6).text = Hit_id, crystal_info_dict["symmetry"]["space_group_name_hm"], str(crystal_info_dict["rcsb_entry_info"]["resolution_combined"][0]) + " Å", \
                            str(crystal_info_dict["cell"]['length_a']) + ", " + str(crystal_info_dict["cell"]['length_b']) + ", " + str(crystal_info_dict["cell"]['length_c']), identities, ", ".join(ligand_name_list), Stoichiometry_info
              
                        except Exception as e:
                            table.cell(PDB_write_status, 0).text, table.cell(PDB_write_status, 1).text, table.cell(PDB_write_status, 2).text, table.cell(PDB_write_status, 3).text, table.cell(PDB_write_status,4).text, table.cell(PDB_write_status, 5).text, table.cell(PDB_write_status, 6).text = Hit_id, crystal_info_dict["symmetry"]["space_group_name_H_M"], str(crystal_info_dict["rcsb_entry_info"]["resolution_combined"][0]) + " Å", \
                            str(crystal_info_dict["cell"]['length_a']) + ", " + str(crystal_info_dict["cell"]['length_b']) + ", " + str(crystal_info_dict["cell"]['length_c']), identities, ", ".join(ligand_name_list), Stoichiometry_info
                            #print("错误信息", e, flush=True)
                            print(Hit_id, flush=True)
                            # input("wait:")

                        break

                    except Exception as e:
                        print("错误信息", e, flush=True)
                        pass
            elif structure_method == "ELECTRON MICROSCOPY":
                for times in range(10):
                    # try:
                        if not os.path.exists(os.path.join(self.work_dir, r'{}.html'.format(Hit_id))):
                            print("  -->第{}尝试".format(times), flush=True)

                            Stoichiometry_response = requests.get(
                                url="https://www.rcsb.org/structure/{0}".format(Hit_id),
                                #proxies=proxies,
                                proxies=None,
                                headers={
                                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
                                    "Referer": "https://www.rcsb.org/",

                                }
                            )
                            if Stoichiometry_response.status_code == 200:
                                with open(os.path.join(self.work_dir, r'{}.html'.format(Hit_id)), "w",
                                          encoding="utf-8") as f_html:
                                    f_html.write(Stoichiometry_response.text)

                                Stoichiometry_soup = BeautifulSoup(Stoichiometry_response.text, "html.parser")

                                Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"carousel-footer"})
                                if Stoichiometry_text is None:
                                    Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"card-footer"})

                                Stoichiometry_info = \
                                    Stoichiometry_text.text.split("Find")[0].split("Global Stoichiometry:")[1].strip()

                            else:
                                # 网络失败
                                Stoichiometry_info = ""
                        else:
                            with open(os.path.join(self.work_dir, r'{}.html'.format(Hit_id)), "r",
                                      encoding="utf-8") as f_html:
                                hitid_html = f_html.read()

                            Stoichiometry_soup = BeautifulSoup(hitid_html, "html.parser")

                            Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"carousel-footer"})
                            if Stoichiometry_text is None:
                                Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"card-footer"})

                            Stoichiometry_info = \
                            Stoichiometry_text.text.split("Find")[0].split("Global Stoichiometry:")[1].strip()

                        try:
                            ligand_name_list = []
                            if crystal_info_dict["rcsb_entry_container_identifiers"]["non_polymer_entity_ids"]:
                                non_polymer_entity_ids = crystal_info_dict["rcsb_entry_container_identifiers"]["non_polymer_entity_ids"]

                                # ligand_name_list = crystal_info_dict["pdbx_vrpt_summary"]["restypes_notchecked_for_bond_angle_geometry"]
                                for non_polymer_entity_id in non_polymer_entity_ids:
                                    ligand_name = self.get_pdb_ligand_name(Hit_id, non_polymer_entity_id)
                                    if ligand_name:
                                        ligand_name_list.append(ligand_name)
                        except:
                            ligand_name_list = []

                        # print(Hit_id,Truncation_length_begin + "-" + Truncation_length_end + " aa" + "({0})".format(identities))
                        try:
                            table.cell(PDB_write_status, 0).text, table.cell(PDB_write_status, 1).text, table.cell(
                                PDB_write_status, 2).text, table.cell(PDB_write_status, 3).text, table.cell(
                                PDB_write_status,
                                4).text, table.cell(
                                PDB_write_status, 5).text, table.cell(PDB_write_status, 6).text = \
                                Hit_id, "", str(
                                    crystal_info_dict["rcsb_entry_info"]["resolution_combined"][0]) + " Å", \
                                "", identities, ", ".join(
                                    ligand_name_list), Stoichiometry_info

                        except Exception as e:
                            print("错误信息", e, flush=True)
                            print(Hit_id, flush=True)
                            # input("wait:")

                        break

                    # except Exception as e:
                    #     print("错误信息", e, flush=True)
                    #     pass

            # PDB 信息写入的记录加1
            PDB_write_status += 1

        ## protein table format
        for cell in table.iter_cells():
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(12)
                # paragraph.line_spacing = 1
                paragraph.font.name = "Times New Roman"
                paragraph.alignment = PP_ALIGN.CENTER
                paragraph.font.color.rgb = RGBColor(0, 0, 0)

    def ppt_expression_and_purification_summary(self, rows=5, columns=2):

        ppt_file = self.ppt_file_handle

        slice = ppt_file.slides.add_slide(ppt_file.slide_layouts[11])
        pptx_protein_title = slice.placeholders[18]
        title = pptx_protein_title.text_frame
        title_paragraph = title.add_paragraph()
        title_run = title.paragraphs[0].add_run()
        title_run.text = "Expression and Purification Summary"
        title_run.font.size = Pt(28)
        title_run.font.bold = True
        title_run.font.color.rgb = RGBColor(146, 50, 34)
        title_run.font.name = "Times New Roman"

        ppt_pdb_summary_table = slice.placeholders[12]

        # protein table
        table = ppt_pdb_summary_table.insert_table(rows, columns).table

        # row height
        for row in table.rows:
            row.height = 150000
            # print(row.height)

        ## protein table header
        table.cell(0, 0).text, table.cell(0, 1).text = \
            "PDB", "Expression and pufification"

        ## protein table format
        for cell in table.iter_cells():
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(12)
                # paragraph.line_spacing = 1
                paragraph.font.name = "Times New Roman"
                paragraph.alignment = PP_ALIGN.CENTER
                paragraph.font.color.rgb = RGBColor(0, 0, 0)

    def ppt_reference(self, rows=2, columns=7):
        # 加载ppt pdb info 信息
        with open(ppt_db_dict_path, "r", encoding="utf-8") as f:
            ppt_db_dict = json.load(f)

        # get pdb id
        for roots,dirs,files in os.walk(self.work_dir):
            for file in files:
                if re.search("-Alignment\.xml$", file):
                    blast_xml_file = os.path.join(roots, file)

        with open(blast_xml_file, "r", encoding="utf-8") as f:
            blast_xml_content = f.read()

        # 解析pdb id, info
        soup = BeautifulSoup(blast_xml_content,"xml")

        Hits_list = soup.find_all(name="Hit")

        # 添加uniprot pdb id
        # for uniprot_pdb_id in self.uniprot_content_dict["pdb_summary_list"]:
        #     if uniprot_pdb_id not in Hits_list:
        #         Hits_list.append(uniprot_pdb_id)
                # print(uniprot_pdb_id)

        PDB_write_status = 1
        for Hit_index, Hit in enumerate(Hits_list):
            # get top 5 PDB
            if PDB_write_status >= 11:
                break

            Hit_id = Hit.find(name="Hit_id").text.split("|")[1]
            Truncation_length_begin = Hit.find(name="Hsp_query-from").text
            Truncation_length_end = Hit.find(name="Hsp_query-to").text
            Hit_identity = Hit.find(name="Hsp_identity").text
            Hit_align_length = Hit.find(name="Hsp_align-len").text
            identities = Truncation_length_begin + "-" + Truncation_length_end + " aa(" + str(round(int(Hit_identity) / int(Hit_align_length) * 100,2)) + "%)"

            print("获取{0}解析方法, pdb url:".format(Hit_id), "https://data.rcsb.org/rest/v1/core/entry/{0}".format(Hit_id), flush=True)

            if os.path.exists(os.path.join(self.work_dir, "crystal_info_{0}.json".format(Hit_id))):
                with open(os.path.join(self.work_dir, "crystal_info_{0}.json".format(Hit_id)), "r", encoding="utf-8") as f:
                    crystal_info_dict = json.load(f)

            else:
                for times in range(10):
                    try:
                        print("  -->第{}尝试".format(times), flush=True)
                        # get crystal_info
                        crystal_response = requests.get(
                            url="https://data.rcsb.org/rest/v1/core/entry/{0}".format(Hit_id),
                            proxies={},
                            headers={
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
                                "Referer": "https://www.rcsb.org/",

                            }
                        )
                        if crystal_response.status_code == 200:
                            break

                    except:
                        time.sleep(5)
                else:
                    # 网络连接失败, 继续下一个
                    continue

                with open(os.path.join(self.work_dir, "crystal_info_{0}.json".format(Hit_id)), "w", encoding="utf-8") as f:
                    f.write(crystal_response.text)

                with open(os.path.join(self.work_dir, "crystal_info_{0}.json".format(Hit_id)), "r", encoding="utf-8") as f:
                    crystal_info_dict = json.load(f)

            # 判断PDB是否为Xray解析出的结构
            structure_method = ""
            try:
                # 新版 RCSB JSON（推荐）
                if crystal_info_dict["rcsb_entry_info"]["experimental_method"]:
                    method = crystal_info_dict["rcsb_entry_info"]["experimental_method"].upper().strip()

                    if "X-RAY" in method:
                        structure_method = "Xray"
                    elif "X-ray" in method:
                        structure_method = "Xray"
                    elif "ELECTRON" in method:
                        structure_method = "ELECTRON MICROSCOPY"
                    elif "EM" in method:
                        structure_method = "ELECTRON MICROSCOPY"  # experimental_method":"EM", 冷冻电镜
                    else:
                        structure_method = method

            except Exception as e:
                try:
                    # 兼容老版 JSON
                    if crystal_info_dict["symmetry"]["space_group_name_H_M"]:
                        structure_method = "Xray"
                except Exception as e:
                    try:
                        # 更老版本
                        if crystal_info_dict["symmetry"]["space_group_name_hm"]:
                            structure_method = "Xray"
                    except Exception as e:
                        try:
                            # 最后再使用 exptl
                            method = crystal_info_dict["exptl"][0]["method"].upper().strip()

                            if "X-RAY" in method:
                                structure_method = "Xray"
                            elif "X-ray" in method:
                                structure_method = "Xray"
                            elif "ELECTRON" in method:
                                structure_method = "ELECTRON MICROSCOPY"
                            elif "EM" in method:
                                structure_method = "ELECTRON MICROSCOPY"
                            else:
                                structure_method = method

                        except Exception as e:
                            print(e)
                            structure_method = "structure method not find"

            for read_i in range(10):
                try:
                    if ppt_db_dict.get(Hit_id):
                        # 导入PPT 模块
                        ppt_app = Dispatch('PowerPoint.Application')
                        print("找到模板", flush=True)
                        # 写PPT
                        # 打开目标PPT
                        object_ppt = ppt_app.Presentations.Open(self.pptx_save_path, WithWindow=False)

                        # 导入模板PPT条信息
                        ppt_db_info_list = ppt_db_dict[Hit_id]

                        for ppt_db_info in ppt_db_info_list:
                            template_pptx_name = ppt_db_info["pptx_name"]
                            template_pptx_path = os.path.join(os.path.dirname(ppt_db_dict_path), template_pptx_name)

                            # 复制模板到当前目录
                            copy_template_pptx_path = os.path.join(self.work_dir, PublicScript.get_current_date() + "_" + template_pptx_name)
                            shutil.copy(template_pptx_path, os.path.join(self.work_dir, copy_template_pptx_path))

                            template_index = ppt_db_info["index"]

                            # 打开模板ppt
                            print("  -->导入模板", template_pptx_name, flush=True)
                            template_ppt = ppt_app.Presentations.Open(copy_template_pptx_path, WithWindow=False)

                            print("  -->复制模板", template_pptx_name, flush=True)
                            template_ppt.Slides(template_index).Copy()
                            print('0000')

                            # 复制PPT
                            print("  -->粘贴模板", template_pptx_name, flush=True)
                            object_ppt.Slides.Paste()
                            print('1111')

                            # 关闭PPt
                            print("  -->关闭模板", template_pptx_name, flush=True)
                            template_ppt.Close()
                            print('2222')

                        print("  -->保存模板", flush=True)
                        object_ppt.SaveAs(self.pptx_save_path)
                        object_ppt.Close()
                        ppt_app.Quit()

                        break
                except Exception as e:
                    # print(traceback.print_exc(), e)
                    time.sleep(5)

            else:
                # 写PPT
                ppt_file = Presentation(self.pptx_save_path)

                slice = ppt_file.slides.add_slide(ppt_file.slide_layouts[12])
                pptx_protein_title = slice.placeholders[18]
                title = pptx_protein_title.text_frame
                title_paragraph = title.add_paragraph()
                title_run = title.paragraphs[0].add_run()
                title_run.text = "Reference for {0}".format(Hit_id)
                title_run.font.size = Pt(28)
                title_run.font.bold = True
                title_run.font.color.rgb = RGBColor(146, 50, 34)
                title_run.font.name = "Times New Roman"

                # # 查看占位符的序号
                # for placeholder in slice.placeholders:
                #     info = placeholder.placeholder_format
                #     print("索引{0},名称{1},类型{2},文本{3}".format(info.idx,placeholder.name,info.type,placeholder.text))

                ppt_reference_table = slice.placeholders[12]
                # ppt_reference_text = slice.placeholders[18]
                ppt_PDBID_text = slice.placeholders[16]
                ppt_proteininfo_text = slice.placeholders[17]

                # protein table
                table = ppt_reference_table.insert_table(rows, columns).table

                # row height
                for row in table.rows:
                    row.height = 150000
                    # print(row.height)

                ## protein table header
                table.cell(0, 0).text, table.cell(0, 1).text, table.cell(0, 2).text, table.cell(0, 3).text, table.cell(0,
                                                                                                                       4).text, table.cell(
                    0, 5).text, table.cell(0, 6).text = \
                    "PDB", "Space Group", "Resolution", "Unit Cell", "Truncation(identities)", "Ligands", "Stoichiometry"
                # table.cell(0, 0).text_frame.paragraphs.font.size = Pt(14)


                # title text
                # ppt_reference_text.text = "Reference for {0}".format(Hit_id)

                print("获取{0}解析参数, pdb url:".format(Hit_id), "https://www.rcsb.org/structure/{0}".format(Hit_id), flush=True)

                if structure_method == "Xray":
                    for times in range(10):
                        try:
                            if not os.path.exists(os.path.join(self.work_dir, r'{}.html'.format(Hit_id))):
                                print("  -->第{}尝试".format(times), flush=True)


                                # get Stoichiometry_info
                                Stoichiometry_response = requests.get(
                                    url="https://www.rcsb.org/structure/{0}".format(Hit_id),
                                    #proxies=proxies,
                                    proxies=None,
                                    headers={
                                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
                                        "Referer": "https://www.rcsb.org/",

                                    }
                                )

                                if Stoichiometry_response.status_code == 200:
                                    with open(os.path.join(self.work_dir, r'{}.html'.format(Hit_id)), "w", encoding="utf-8") as f_html:
                                        f_html.write(Stoichiometry_response.text)

                                    Stoichiometry_soup = BeautifulSoup(Stoichiometry_response.text, "html.parser")

                                    # get Stoichiometry_info
                                    Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"carousel-footer"})
                                    if Stoichiometry_text is None:
                                        Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"card-footer"})

                                    Stoichiometry_info = \
                                    Stoichiometry_text.text.split("Find")[0].split("Global Stoichiometry:")[1].strip()

                                else:
                                    # 网络失败
                                    Stoichiometry_info = ""

                            else:
                                with open(os.path.join(self.work_dir, r'{}.html'.format(Hit_id)), "r", encoding="utf-8") as f_html:
                                    hitid_html = f_html.read()

                                Stoichiometry_soup = BeautifulSoup(hitid_html, "html.parser")

                                # get Stoichiometry_info
                                Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"carousel-footer"})
                                if Stoichiometry_text is None:
                                    Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"card-footer"})
                                Stoichiometry_info = Stoichiometry_text.text.split("Find")[0].split("Global Stoichiometry:")[1].strip()

                            # get image pic
                            pic_response = requests.get(
                                url="https://cdn.rcsb.org/images/structures/{1}/{0}/{0}_assembly-1.jpeg".format(Hit_id.lower(),Hit_id.lower()[1:3]),
                                #proxies=proxies,
                                proxies=None,
                                headers={
                                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
                                    "Referer": "https://www.rcsb.org/",

                                }
                            )

                            if pic_response.status_code == 200:
                                with open(os.path.join(self.work_dir,"{0}.jpeg".format(Hit_id)),"wb") as f:
                                    f.write(pic_response.content)

                            try:
                                ligand_name_list = []


                                if crystal_info_dict["rcsb_entry_container_identifiers"]["non_polymer_entity_ids"]:
                                    non_polymer_entity_ids = crystal_info_dict["rcsb_entry_container_identifiers"][
                                        "non_polymer_entity_ids"]

                                    # ligand_name_list = crystal_info_dict["pdbx_vrpt_summary"]["restypes_notchecked_for_bond_angle_geometry"]
                                    for non_polymer_entity_id in non_polymer_entity_ids:
                                        ligand_name = self.get_pdb_ligand_name(Hit_id, non_polymer_entity_id)
                                        if ligand_name:
                                            ligand_name_list.append(ligand_name)

                            except:
                                ligand_name_list = []

                            # print(Hit_id,Truncation_length_begin + "-" + Truncation_length_end + " aa" + "({0})".format(identities))
                            try:
                                table.cell(1, 0).text, table.cell(1, 1).text, table.cell(1, 2).text, table.cell(1, 3).text, table.cell(1,
                                                                                                                                       4).text, table.cell(
                                    1, 5).text, table.cell(1, 6).text = \
                                    Hit_id, crystal_info_dict["symmetry"]["space_group_name_hm"], str(crystal_info_dict["rcsb_entry_info"]["resolution_combined"][0]) + " Å", \
                                str(crystal_info_dict["cell"]['length_a']) + ", " + str(crystal_info_dict["cell"]['length_b']) + ", " + str(crystal_info_dict["cell"]['length_c']), identities, ", ".join(ligand_name_list), Stoichiometry_info
                            except Exception as e:
                                table.cell(1, 0).text, table.cell(1, 1).text, table.cell(1, 2).text, table.cell(1, 3).text, table.cell(1,
                                                                                                                                                                   4).text, table.cell(
                                                                1, 5).text, table.cell(1, 6).text = \
                                                                Hit_id, crystal_info_dict["symmetry"]["space_group_name_H_M"], str(crystal_info_dict["rcsb_entry_info"]["resolution_combined"][0]) + " Å", \
                                                            str(crystal_info_dict["cell"]['length_a']) + ", " + str(crystal_info_dict["cell"]['length_b']) + ", " + str(crystal_info_dict["cell"]['length_c']), identities, ", ".join(ligand_name_list), Stoichiometry_info
                                


                            ## protein table format
                            for cell in table.iter_cells():
                                for paragraph in cell.text_frame.paragraphs:
                                    paragraph.font.size = Pt(12)
                                    # paragraph.line_spacing = 1
                                    paragraph.font.name = "Times New Roman"
                                    paragraph.alignment = PP_ALIGN.CENTER
                                    paragraph.font.color.rgb = RGBColor(0, 0, 0)


                            # insert pic
                            left = Cm(1.6)
                            top = Cm(7)
                            height = Cm(6.5)
                            width = Cm(6.5)
                            pic = slice.shapes.add_picture(os.path.join(self.work_dir,"{0}.jpeg".format(Hit_id)), left, top, width=width,height=height)

                            # add pdb info
                            pdbid_info_text_frame = ppt_PDBID_text.text_frame
                            pdbid_info_paragraph = pdbid_info_text_frame.add_paragraph()
                            ## pdbid info
                            pdbid_name = pdbid_info_text_frame.paragraphs[0].add_run()
                            pdbid_name.text = "PDB: {0}".format(Hit_id)
                            pdbid_name.font.bold = True
                            pdbid_name.font.color.rgb = RGBColor(255, 0, 0)

                            ## reference info
                            pdbid_reference_title_name = pdbid_info_paragraph.add_run()
                            pdbid_reference_title_name.text = crystal_info_dict["rcsb_primary_citation"]["title"] + "\n"

                            pdbid_reference_publish_info = pdbid_info_paragraph.add_run()

                            if crystal_info_dict["rcsb_primary_citation"].get("year",None):
                                publish_year = crystal_info_dict["rcsb_primary_citation"]["year"]
                            else:
                                publish_year = ""

                            if crystal_info_dict["rcsb_primary_citation"].get("journal_abbrev", None):
                                publish_journal_abbrev = crystal_info_dict["rcsb_primary_citation"]["journal_abbrev"]
                            else:
                                publish_journal_abbrev = ""

                            if crystal_info_dict["rcsb_primary_citation"].get("journal_volume", None):
                                publish_journal_volume = crystal_info_dict["rcsb_primary_citation"]["journal_volume"]
                            else:
                                publish_journal_volume = ""

                            if crystal_info_dict["rcsb_primary_citation"].get("page_first",None):
                                # print(Hit_id)
                                publish_page_first = crystal_info_dict["rcsb_primary_citation"]["page_first"]

                                if crystal_info_dict["rcsb_primary_citation"].get("page_last", None):
                                    publish_page_last = crystal_info_dict["rcsb_primary_citation"]["page_last"]
                                else:
                                    publish_page_last = ""

                                pdbid_reference_publish_info.text = "({0}) {1} {2}: {3}-{4}".format(publish_year,publish_journal_abbrev,publish_journal_volume,publish_page_first,publish_page_last)

                            else:

                                pdbid_reference_publish_info.text = "({0}) {1} {2}".format(publish_year,publish_journal_abbrev,publish_journal_volume)
                            # pdbid_reference_title.font.bold = True

                            # 分析文献信息
                            hitid_literature_path = os.path.join(rcsb_literature_dir, "{}.pdf".format(Hit_id.upper()))
                            local_literature_path = os.path.join(self.work_dir, "{}.pdf".format(Hit_id.upper()))
                            print("文献信息:", "{}.pdf".format(Hit_id.upper()), os.path.exists(hitid_literature_path), flush=True, end="\r")
                            if os.path.exists(hitid_literature_path):
                                shutil.copy(hitid_literature_path, local_literature_path)

                                # 上传pdf
                                chatpdf_input_pdf_path = os.path.join(chatpdf_input_dir, "{}.pdf".format(Hit_id.upper()))
                                shutil.copy(local_literature_path, chatpdf_input_pdf_path)

                                # 等待解析pdf
                                chatpdf_output_json_path = os.path.join(chatpdf_output_dir, "{}_biortus.json".format(Hit_id.upper()))
                                local_output_json_path = os.path.join(self.work_dir, "{}_biortus.json".format(Hit_id.upper()))
                                print("等待解析pdf", flush=True)
                                for pdf_wait in range(30):
                                    print("预计等待剩余时间: {} min".format(30 - pdf_wait), flush=True)
                                    if os.path.exists(chatpdf_output_json_path):
                                        shutil.copy(chatpdf_output_json_path, local_output_json_path)

                                        with open(local_output_json_path, "r", encoding="utf-8") as f_json:
                                            proteininfo_dict = json.load(f_json)

                                        # 输出蛋白信息
                                        print("***********生成文献信息***********", flush=True)
                                        protein_info_text_frame = ppt_proteininfo_text.text_frame
                                        protein_info_paragraph = protein_info_text_frame.add_paragraph()
                                        ## protein name
                                        protein_name = protein_info_text_frame.paragraphs[0].add_run()
                                        protein_name.text = "Expression: "
                                        protein_name.font.bold = True

                                        # protein_name_info = protein_info_text_frame.paragraphs[0].add_run()
                                        # protein_name_info.text = self.uniprot_content_dict["protein_recommended_name"]

                                        ## protein vector name
                                        protein_vector_name = protein_info_paragraph.add_run()
                                        protein_vector_name.text = "  Vector: "
                                        protein_vector_name.font.bold = True

                                        protein_vector_name_info = protein_info_paragraph.add_run()
                                        protein_vector_name_info.text = proteininfo_dict["protein vector"]["answer"].strip() + "\n"

                                        ## protein tag name
                                        protein_tag_name = protein_info_paragraph.add_run()
                                        protein_tag_name.text = "  Tag: "
                                        protein_tag_name.font.bold = True

                                        protein_tag_name_info = protein_info_paragraph.add_run()
                                        protein_tag_name_info.text = proteininfo_dict["protein tag"][
                                                                            "answer"].strip() + "\n"

                                        ## protein sequence name
                                        protein_sequence_name = protein_info_paragraph.add_run()
                                        protein_sequence_name.text = "  Sequence: "
                                        protein_sequence_name.font.bold = True

                                        protein_sequence_name_info = protein_info_paragraph.add_run()
                                        protein_sequence_name_info.text = proteininfo_dict["protein sequence"][
                                                                            "answer"].strip() + "\n"

                                        ## protein host name
                                        protein_host_name = protein_info_paragraph.add_run()
                                        protein_host_name.text = "  Host: "
                                        protein_host_name.font.bold = True

                                        protein_host_name_info = protein_info_paragraph.add_run()
                                        protein_host_name_info.text = proteininfo_dict["protein host"][
                                                                            "answer"].strip() + "\n"

                                        ## protein purification name
                                        protein_purification_name = protein_info_paragraph.add_run()
                                        protein_purification_name.text = "Purification: \n"
                                        protein_purification_name.font.bold = True

                                        protein_purification_name = protein_info_paragraph.add_run()
                                        protein_purification_name.text = "  Steps: "
                                        protein_purification_name.font.bold = True

                                        protein_purification_name_info = protein_info_paragraph.add_run()
                                        protein_purification_name_info.text = proteininfo_dict["protein purification steps"][
                                                                            "answer"].strip() + "\n"

                                        ## protein buffer name
                                        protein_buffer_name = protein_info_paragraph.add_run()
                                        protein_buffer_name.text = "  Buffer: "
                                        protein_buffer_name.font.bold = True

                                        protein_buffer_name_info = protein_info_paragraph.add_run()
                                        protein_buffer_name_info.text = proteininfo_dict["protein purification buffer"][
                                                                            "answer"].strip() + "\n"

                                        ## protein crystallization  name
                                        protein_crystallization_name = protein_info_paragraph.add_run()
                                        protein_crystallization_name.text = "Crystallization: \n"
                                        protein_crystallization_name.font.bold = True

                                        protein_crystallization_name = protein_info_paragraph.add_run()
                                        protein_crystallization_name.text = "  Steps: "
                                        protein_crystallization_name.font.bold = True

                                        protein_crystallization_name_info = protein_info_paragraph.add_run()
                                        protein_crystallization_name_info.text = proteininfo_dict["protein crystallization steps"][
                                                                            "answer"].strip() + "\n"

                                        ## protein condition name
                                        crystallization_condition_name = protein_info_paragraph.add_run()
                                        crystallization_condition_name.text = "  Condition: "
                                        crystallization_condition_name.font.bold = True

                                        crystallization_condition_name_info = protein_info_paragraph.add_run()
                                        crystallization_condition_name_info.text = proteininfo_dict["protein crystallization condition"][
                                                                            "answer"].strip() + "\n"

                                        break

                                    time.sleep(60)
                            else:
                                placeholder = slice.placeholders[17].element
                                placeholder.getparent().remove(placeholder)
                            ppt_file.save(self.pptx_save_path)

                            '''
                            插入文献
                            '''
                            for leterature_i in range(5):
                                try:
                                    # hitid_literature_path = os.path.join(rcsb_literature_dir, "{}.pdf".format(Hit_id.upper()))
                                    # local_literature_path = os.path.join(self.work_dir, "{}.pdf".format(Hit_id.upper()))
                                    if os.path.exists(local_literature_path):
                                        # shutil.copy(hitid_literature_path, local_literature_path)

                                        # 导入PPT 模块
                                        ppt_app = Dispatch('PowerPoint.Application')
                                        ppt_template = ppt_app.Presentations.Open(self.pptx_save_path, WithWindow=False)

                                        # pdf初始插入位置
                                        pdf_left = 650
                                        pdf_top = 200
                                        pdf_width = 70
                                        pdf_height = 30

                                        # print(insert_ppt_index)
                                        ppt_template.Slides(ppt_template.Slides.Count).Shapes.AddOLEObject(pdf_left, pdf_top, pdf_width,
                                                                                                  pdf_height,
                                                                                                  FileName=local_literature_path,
                                                                                                  IconLabel=Hit_id,
                                                                                                  DisplayAsIcon=False)

                                        # 保存目标PPT
                                        ppt_template.SaveAs(self.pptx_save_path)

                                        ppt_template.Close()
                                        ppt_app.Quit()

                                        break

                                except Exception as e:
                                    print("插入文献出错,", e, flush=True)

                            break

                        except Exception as e:
                            print("错误信息", e, flush=True)
                            pass

                elif structure_method == "ELECTRON MICROSCOPY":
                    for times in range(10):
                        try:
                            if not os.path.exists(os.path.join(self.work_dir, r'{}.html'.format(Hit_id))):
                                print("  -->第{}尝试".format(times), flush=True)

                                # get Stoichiometry_info
                                Stoichiometry_response = requests.get(
                                    url="https://www.rcsb.org/structure/{0}".format(Hit_id),
                                    #proxies=proxies,
                                    proxies=None,
                                    headers={
                                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
                                        "Referer": "https://www.rcsb.org/",

                                    }
                                )

                                if Stoichiometry_response.status_code == 200:
                                    with open(os.path.join(self.work_dir, r'{}.html'.format(Hit_id)), "w",
                                              encoding="utf-8") as f_html:
                                        f_html.write(Stoichiometry_response.text)

                                    Stoichiometry_soup = BeautifulSoup(Stoichiometry_response.text, "html.parser")

                                    Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"carousel-footer"})
                                    if Stoichiometry_text is None:
                                        Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"card-footer"})

                                    Stoichiometry_info = \
                                        Stoichiometry_text.text.split("Find")[0].split("Global Stoichiometry:")[
                                            1].strip()

                                else:
                                    # 网络失败
                                    Stoichiometry_info = ""

                            else:
                                with open(os.path.join(self.work_dir, r'{}.html'.format(Hit_id)), "r",
                                          encoding="utf-8") as f_html:
                                    hitid_html = f_html.read()

                                Stoichiometry_soup = BeautifulSoup(hitid_html, "html.parser")

                                # get Stoichiometry_info
                                Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"carousel-footer"})
                                if Stoichiometry_text is None:
                                    Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"card-footer"})

                                Stoichiometry_info = \
                                Stoichiometry_text.text.split("Find")[0].split("Global Stoichiometry:")[1].strip()

                            # get image pic
                            pic_response = requests.get(
                                url="https://cdn.rcsb.org/images/structures/{1}/{0}/{0}_assembly-1.jpeg".format(
                                    Hit_id.lower(), Hit_id.lower()[1:3]),
                                #proxies=proxies,
                                proxies=None,
                                headers={
                                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
                                    "Referer": "https://www.rcsb.org/",

                                }
                            )

                            if pic_response.status_code == 200:
                                with open(os.path.join(self.work_dir, "{0}.jpeg".format(Hit_id)), "wb") as f:
                                    f.write(pic_response.content)

                            try:
                                ligand_name_list = []


                                if crystal_info_dict["rcsb_entry_container_identifiers"]["non_polymer_entity_ids"]:
                                    non_polymer_entity_ids = crystal_info_dict["rcsb_entry_container_identifiers"][
                                        "non_polymer_entity_ids"]

                                    # ligand_name_list = crystal_info_dict["pdbx_vrpt_summary"]["restypes_notchecked_for_bond_angle_geometry"]
                                    for non_polymer_entity_id in non_polymer_entity_ids:
                                        ligand_name = self.get_pdb_ligand_name(Hit_id, non_polymer_entity_id)
                                        if  ligand_name:
                                            ligand_name_list.append(ligand_name)
                            except:
                                ligand_name_list = []

                            # print(Hit_id,Truncation_length_begin + "-" + Truncation_length_end + " aa" + "({0})".format(identities))
                            table.cell(1, 0).text, table.cell(1, 1).text, table.cell(1, 2).text, table.cell(1,3).text, table.cell(1,4).text, table.cell(1, 5).text, table.cell(1, 6).text = \
                                Hit_id, "", str(
                                    crystal_info_dict["rcsb_entry_info"]["resolution_combined"][0]) + " Å", \
                                "", identities, ", ".join(
                                    ligand_name_list), Stoichiometry_info

                            ## protein table format
                            for cell in table.iter_cells():
                                for paragraph in cell.text_frame.paragraphs:
                                    paragraph.font.size = Pt(12)
                                    # paragraph.line_spacing = 1
                                    paragraph.font.name = "Times New Roman"
                                    paragraph.alignment = PP_ALIGN.CENTER
                                    paragraph.font.color.rgb = RGBColor(0, 0, 0)

                            # insert pic
                            left = Cm(1.6)
                            top = Cm(7)
                            height = Cm(6.5)
                            width = Cm(6.5)
                            pic = slice.shapes.add_picture(os.path.join(self.work_dir, "{0}.jpeg".format(Hit_id)), left,
                                                           top, width=width, height=height)

                            # add pdb info
                            pdbid_info_text_frame = ppt_PDBID_text.text_frame
                            pdbid_info_paragraph = pdbid_info_text_frame.add_paragraph()
                            ## pdbid info
                            pdbid_name = pdbid_info_text_frame.paragraphs[0].add_run()
                            pdbid_name.text = "PDB: {0}".format(Hit_id)
                            pdbid_name.font.bold = True
                            pdbid_name.font.color.rgb = RGBColor(255, 0, 0)

                            ## reference info
                            pdbid_reference_title_name = pdbid_info_paragraph.add_run()
                            pdbid_reference_title_name.text = crystal_info_dict["rcsb_primary_citation"]["title"] + "\n"

                            pdbid_reference_publish_info = pdbid_info_paragraph.add_run()

                            if crystal_info_dict["rcsb_primary_citation"].get("year", None):
                                publish_year = crystal_info_dict["rcsb_primary_citation"]["year"]
                            else:
                                publish_year = ""

                            if crystal_info_dict["rcsb_primary_citation"].get("journal_abbrev", None):
                                publish_journal_abbrev = crystal_info_dict["rcsb_primary_citation"]["journal_abbrev"]
                            else:
                                publish_journal_abbrev = ""

                            if crystal_info_dict["rcsb_primary_citation"].get("journal_volume", None):
                                publish_journal_volume = crystal_info_dict["rcsb_primary_citation"]["journal_volume"]
                            else:
                                publish_journal_volume = ""

                            if crystal_info_dict["rcsb_primary_citation"].get("page_first", None):
                                # print(Hit_id)
                                publish_page_first = crystal_info_dict["rcsb_primary_citation"]["page_first"]

                                if crystal_info_dict["rcsb_primary_citation"].get("page_last", None):
                                    publish_page_last = crystal_info_dict["rcsb_primary_citation"]["page_last"]
                                else:
                                    publish_page_last = ""

                                pdbid_reference_publish_info.text = "({0}) {1} {2}: {3}-{4}".format(publish_year,
                                                                                                    publish_journal_abbrev,
                                                                                                    publish_journal_volume,
                                                                                                    publish_page_first,
                                                                                                    publish_page_last)

                            else:

                                pdbid_reference_publish_info.text = "({0}) {1} {2}".format(publish_year,
                                                                                           publish_journal_abbrev,
                                                                                           publish_journal_volume)
                            # pdbid_reference_title.font.bold = True

                            # 分析文献信息
                            hitid_literature_path = os.path.join(rcsb_literature_dir, "{}.pdf".format(Hit_id.upper()))
                            local_literature_path = os.path.join(self.work_dir, "{}.pdf".format(Hit_id.upper()))
                            print("文献信息:", "{}.pdf".format(Hit_id.upper()), os.path.exists(hitid_literature_path), flush=True)
                            if os.path.exists(hitid_literature_path):
                                shutil.copy(hitid_literature_path, local_literature_path)

                                # 上传pdf
                                chatpdf_input_pdf_path = os.path.join(chatpdf_input_dir,
                                                                      "{}.pdf".format(Hit_id.upper()))
                                shutil.copy(local_literature_path, chatpdf_input_pdf_path)

                                # 等待解析pdf
                                chatpdf_output_json_path = os.path.join(chatpdf_output_dir,
                                                                        "{}_biortus.json".format(Hit_id.upper()))
                                local_output_json_path = os.path.join(self.work_dir,
                                                                      "{}_biortus.json".format(Hit_id.upper()))
                                print("等待解析pdf", flush=True)
                                for pdf_wait in range(30):
                                    print("预计等待剩余时间: {} min".format(30 - pdf_wait), flush=True)
                                    if os.path.exists(chatpdf_output_json_path):
                                        shutil.copy(chatpdf_output_json_path, local_output_json_path)

                                        with open(local_output_json_path, "r", encoding="utf-8") as f_json:
                                            proteininfo_dict = json.load(f_json)

                                        print("***********生成文献信息***********", flush=True)
                                        # 输出蛋白信息
                                        protein_info_text_frame = ppt_proteininfo_text.text_frame
                                        protein_info_paragraph = protein_info_text_frame.add_paragraph()
                                        ## protein name
                                        protein_name = protein_info_text_frame.paragraphs[0].add_run()
                                        protein_name.text = "Expression: "
                                        protein_name.font.bold = True

                                        # protein_name_info = protein_info_text_frame.paragraphs[0].add_run()
                                        # protein_name_info.text = self.uniprot_content_dict["protein_recommended_name"]

                                        ## protein vector name
                                        protein_vector_name = protein_info_paragraph.add_run()
                                        protein_vector_name.text = "  Vector: "
                                        protein_vector_name.font.bold = True

                                        protein_vector_name_info = protein_info_paragraph.add_run()
                                        protein_vector_name_info.text = proteininfo_dict["protein vector"][
                                                                            "answer"].strip() + "\n"

                                        ## protein tag name
                                        protein_tag_name = protein_info_paragraph.add_run()
                                        protein_tag_name.text = "  Tag: "
                                        protein_tag_name.font.bold = True

                                        protein_tag_name_info = protein_info_paragraph.add_run()
                                        protein_tag_name_info.text = proteininfo_dict["protein tag"][
                                                                         "answer"].strip() + "\n"

                                        ## protein sequence name
                                        protein_sequence_name = protein_info_paragraph.add_run()
                                        protein_sequence_name.text = "  Sequence: "
                                        protein_sequence_name.font.bold = True

                                        protein_sequence_name_info = protein_info_paragraph.add_run()
                                        protein_sequence_name_info.text = proteininfo_dict["protein sequence"][
                                                                              "answer"].strip() + "\n"

                                        ## protein host name
                                        protein_host_name = protein_info_paragraph.add_run()
                                        protein_host_name.text = "  Host: "
                                        protein_host_name.font.bold = True

                                        protein_host_name_info = protein_info_paragraph.add_run()
                                        protein_host_name_info.text = proteininfo_dict["protein host"][
                                                                          "answer"].strip() + "\n"

                                        ## protein purification name
                                        protein_purification_name = protein_info_paragraph.add_run()
                                        protein_purification_name.text = "Purification: \n"
                                        protein_purification_name.font.bold = True

                                        protein_purification_name = protein_info_paragraph.add_run()
                                        protein_purification_name.text = "  Steps: "
                                        protein_purification_name.font.bold = True

                                        protein_purification_name_info = protein_info_paragraph.add_run()
                                        protein_purification_name_info.text = \
                                        proteininfo_dict["protein purification steps"][
                                            "answer"].strip() + "\n"

                                        ## protein buffer name
                                        protein_buffer_name = protein_info_paragraph.add_run()
                                        protein_buffer_name.text = "  Buffer: "
                                        protein_buffer_name.font.bold = True

                                        protein_buffer_name_info = protein_info_paragraph.add_run()
                                        protein_buffer_name_info.text = proteininfo_dict["protein purification buffer"][
                                                                            "answer"].strip() + "\n"

                                        ## protein crystallization  name
                                        protein_crystallization_name = protein_info_paragraph.add_run()
                                        protein_crystallization_name.text = "Crystallization: \n"
                                        protein_crystallization_name.font.bold = True

                                        protein_crystallization_name = protein_info_paragraph.add_run()
                                        protein_crystallization_name.text = "  Steps: "
                                        protein_crystallization_name.font.bold = True

                                        protein_crystallization_name_info = protein_info_paragraph.add_run()
                                        protein_crystallization_name_info.text = \
                                        proteininfo_dict["protein crystallization steps"][
                                            "answer"].strip() + "\n"

                                        ## protein condition name
                                        crystallization_condition_name = protein_info_paragraph.add_run()
                                        crystallization_condition_name.text = "  Condition: "
                                        crystallization_condition_name.font.bold = True

                                        crystallization_condition_name_info = protein_info_paragraph.add_run()
                                        crystallization_condition_name_info.text = \
                                        proteininfo_dict["protein crystallization condition"][
                                            "answer"].strip() + "\n"

                                        break

                                    time.sleep(60)
                            else:
                                placeholder = slice.placeholders[17].element
                                placeholder.getparent().remove(placeholder)
                            ppt_file.save(self.pptx_save_path)

                            '''
                            插入文献
                            '''
                            for leterature_i in range(5):
                                try:
                                    # hitid_literature_path = os.path.join(rcsb_literature_dir, "{}.pdf".format(Hit_id.upper()))
                                    # local_literature_path = os.path.join(self.work_dir, "{}.pdf".format(Hit_id.upper()))
                                    if os.path.exists(local_literature_path):
                                        # shutil.copy(hitid_literature_path, local_literature_path)

                                        # 导入PPT 模块
                                        ppt_app = Dispatch('PowerPoint.Application')
                                        ppt_template = ppt_app.Presentations.Open(self.pptx_save_path, WithWindow=False)

                                        # pdf初始插入位置
                                        pdf_left = 650
                                        pdf_top = 200
                                        pdf_width = 70
                                        pdf_height = 30

                                        # print(insert_ppt_index)
                                        ppt_template.Slides(ppt_template.Slides.Count).Shapes.AddOLEObject(pdf_left, pdf_top,
                                                                                                           pdf_width,
                                                                                                           pdf_height,
                                                                                                           FileName=local_literature_path,
                                                                                                           IconLabel=Hit_id,
                                                                                                           DisplayAsIcon=False)

                                        # 保存目标PPT
                                        ppt_template.SaveAs(self.pptx_save_path)

                                        ppt_template.Close()
                                        ppt_app.Quit()

                                        break

                                except Exception as e:
                                    print("插入文献出错,", e, flush=True)

                            break

                        except Exception as e:
                            print("错误信息", e, flush=True)
                            pass

                else:
                    # 既不是Xray也不是EM(如NMR/Multiple methods/方法未识别), 无对应Reference模板分支
                    print("  -->警告: {0} 解析方法为 {1}, 无对应Reference模板分支(Xray/EM), 跳过该PDB".format(Hit_id, structure_method), flush=True)

            # PDB 写入的信息记录加1
            PDB_write_status += 1

    def biortus_blast(self, rows=11, columns=3):
        ppt_file = self.ppt_file_handle

        # 生成序列文件
        protein_seq_name = "{}.fasta".format(self.uniprot_id)
        protein_seq_path = os.path.join(self.work_dir, protein_seq_name)

        with open(protein_seq_path, "w", encoding="utf-8") as f:
            f.write(">{0}\n{1}".format(self.uniprot_id, self.protein_seq))

        # 上传至服务器, 生成blast结果
        # shared_blast_seq_path = os.path.join(shared_blast_seq_dir, protein_seq_name)
        if os.path.exists(protein_seq_path):
            # shutil.copy(protein_seq_path, shared_blast_seq_path)
            self.get_biortus_blast(protein_seq_path)

            # 等待服务器回传Blast 结果
            protein_xml_name = protein_seq_name.replace(".fasta", "_biortus.xml")
            protein_xml_path = os.path.join(self.work_dir, protein_xml_name)
            shared_blast_xml_path = os.path.join(shared_blast_xml_pir, protein_xml_name)
            for times in range(30):

                if os.path.exists(shared_blast_xml_path):
                    shutil.copy(shared_blast_xml_path, protein_xml_path)
                    break

                time.sleep(10)

        # 解析xml
        if os.path.exists(protein_xml_path):
            # 导入PPT
            slice = ppt_file.slides.add_slide(ppt_file.slide_layouts[18])
            pptx_protein_title = slice.placeholders[18]
            title = pptx_protein_title.text_frame
            title_paragraph = title.add_paragraph()
            title_run = title.paragraphs[0].add_run()
            title_run.text = "Biortus Plasmid Info"
            title_run.font.size = Pt(28)
            title_run.font.bold = True
            title_run.font.color.rgb = RGBColor(146, 50, 34)
            title_run.font.name = "Times New Roman"

            ppt_pdb_summary_table = slice.placeholders[12]

            # protein table
            table = ppt_pdb_summary_table.insert_table(rows, columns).table

            # row height
            for row in table.rows:
                row.height = 150000
                # print(row.height)

            ## protein table header
            table.cell(0, 0).text, table.cell(0, 1).text, table.cell(0, 2).text = "Biortus Plasmid ID", "Biortus Plasmid Name", "Truncation(identities)"

            with open(protein_xml_path, "r", encoding="utf-8") as f:
                blast_xml_content = f.read()

            soup = BeautifulSoup(blast_xml_content, "xml")

            Hits_list = soup.find_all(name="Hit")

            PDB_write_status = 1
            for Hit_index, Hit in enumerate(Hits_list):
                # get top 5 PDB
                if PDB_write_status >= 11:
                    break

                Hit_id = Hit.find(name="Hit_def").text
                Truncation_length_begin = Hit.find(name="Hsp_query-from").text
                Truncation_length_end = Hit.find(name="Hsp_query-to").text
                Hit_identity = Hit.find(name="Hsp_identity").text
                Hit_align_length = Hit.find(name="Hsp_align-len").text
                identities = Truncation_length_begin + "-" + Truncation_length_end + " aa(" + str(
                    round(int(Hit_identity) / int(Hit_align_length) * 100, 2)) + "%)"

                # 获取质粒编号
                plasmid_id_num = ""
                if re.search("(\d+)#|#(\d+)|#\s+(\d+)", Hit_id):
                    if re.search("(\d+)#|#(\d+)|#\s+(\d+)", Hit_id).group(1):
                        plasmid_id_num = re.search("(\d+)#|#(\d+)|#\s+(\d+)", Hit_id).group(1)
                    elif re.search("(\d+)#|#(\d+)|#\s+(\d+)", Hit_id).group(2):
                        plasmid_id_num = re.search("(\d+)#|#(\d+)|#\s+(\d+)", Hit_id).group(2)
                    elif re.search("(\d+)#|#(\d+)|#\s+(\d+)", Hit_id).group(3):
                        plasmid_id_num = re.search("(\d+)#|#(\d+)|#\s+(\d+)", Hit_id).group(3)

                if plasmid_id_num:
                    table.cell(PDB_write_status, 0).text, table.cell(PDB_write_status, 1).text, table.cell(PDB_write_status, 2).text = plasmid_id_num, Hit_id, identities
                else:
                    table.cell(PDB_write_status, 0).text, table.cell(PDB_write_status, 1).text, table.cell(PDB_write_status, 2).text = Hit_id, Hit_id, identities

                PDB_write_status += 1

            ## protein table format
            for cell in table.iter_cells():
                for paragraph in cell.text_frame.paragraphs:
                    paragraph.font.size = Pt(12)
                    # paragraph.line_spacing = 1
                    paragraph.font.name = "Times New Roman"
                    paragraph.alignment = PP_ALIGN.CENTER
                    paragraph.font.color.rgb = RGBColor(0, 0, 0)
        else:
            # 导入PPT
            slice = ppt_file.slides.add_slide(ppt_file.slide_layouts[18])
            pptx_protein_title = slice.placeholders[18]
            title = pptx_protein_title.text_frame
            title_paragraph = title.add_paragraph()
            title_run = title.paragraphs[0].add_run()
            title_run.text = "Biortus Plasmid Info"
            title_run.font.size = Pt(28)
            title_run.font.bold = True
            title_run.font.color.rgb = RGBColor(146, 50, 34)
            title_run.font.name = "Times New Roman"

        # 商品蛋白模板所在的初始PPT页数
        self.commercial_ppt_start_index = len(ppt_file.slides) + 1 - 4 # 4张模板未删除
        print('商品蛋白模板所在的初始PPT页数', self.commercial_ppt_start_index)
        # print("商品蛋白模板所在的初始PPT页数", self.commercial_ppt_start_index)

    def insert_obj(self):
        pass

    def plasmids_design(self):
        ppt_handle = self.ppt_file_handle

        for ppt_index, slide in enumerate(ppt_handle.slides):
            # print("ppt index", ppt_index)
            for shape in slide.shapes:

                # 判断是否为文本框
                text_shape = self.iter_shape(shape)
                if text_shape:
                    # text = text_shape.text_frame.text
                    try:
                        text = text_shape.text_frame
                    except:
                        continue

                    for paragraph in text.paragraphs:
                        for run in paragraph.runs:
                            # print(run.text)
                            # input("w:")
                            if run.text.strip() == "Truncations_Text":
                                # print("ppt index", ppt_index)
                                domain_text = ""
                                for domain_info_list in self.domain_list:
                                    merge_domain_list = []
                                    for domain_info in domain_info_list:
                                        try:
                                            merge_domain_list.extend(
                                                [int(domain_info.split("-")[0]), int(domain_info.split("-")[1])])
                                        except ValueError:
                                            print(f"Warning: 跳过无效 domain_info: '{domain_info}'")

                                    domain_start = min(merge_domain_list)
                                    domain_end = max(merge_domain_list)
                                    domain_info = "{0}-{1}".format(domain_start, domain_end)  # 更新连接后的domain信息

                                    # 只统计长度占总长度10%以上的结构域(新算法不需要这么做)
                                    domain_length_percentage = (domain_end - domain_start) / len(self.protein_seq)
                                    if domain_length_percentage >= self.domain_length_threhold or True:
                                        domain_text += domain_info + ","    # 输出更新连接后的domain信息

                                    domain_text = re.sub(",$", "\n", domain_text)

                                run.text = domain_text

                            if run.text.strip() == "Plasmids_Design_Text":
                                # print("ppt index", ppt_index)
                                plasminds_design_text = ""

                                for domain_info_list in self.domain_list:
                                    merge_domain_list = []
                                    for domain_info in domain_info_list:
                                        try:
                                            merge_domain_list.extend(
                                            [int(domain_info.split("-")[0]), int(domain_info.split("-")[1])])
                                        except ValueError:
                                            print(f"Warning: 跳过无效 domain_info: '{domain_info}'")

                                    domain_start = min(merge_domain_list)
                                    domain_end = max(merge_domain_list)
                                    domain_info = "{0}-{1}".format(domain_start, domain_end)  # 更新连接后的domain信息

                                    # 只统计长度占总长度10%以上的结构域(新算法不需要这么做)
                                    domain_length_percentage = (domain_end - domain_start) / len(self.protein_seq)
                                    if domain_length_percentage >= self.domain_length_threhold or True:
                                        for plasmids_name in self.plasmids_name_list:
                                            # 输出更新连接后的domain信息
                                            #plasminds_design_text += "8His-StrepII-8His-Sumo*-{0}({1}) in {2}\n".format(self.gene_name, domain_info, plasmids_name)
                                            plasminds_design_text += "8His-StrepII-TEV-GG-{0}({1}) in {2}\n".format(
                                                self.gene_name, domain_info, plasmids_name)

                                run.text = plasminds_design_text

    def  generate_plasmid_Word(self):
        # 插入 plasmid word ppt
        # ppt_handle = self.ppt_file_handle
        # slice = ppt_handle.slides.add_slide(ppt_handle.slide_layouts[19])

        """
        生成质粒信息文档
        """
        # 查找模板
        # 保存shared_blast_xml_path字典, 用于从服务器取回文件
        shared_blast_xml_path_dict = {}

        for domain_info_list in self.domain_list:
            merge_domain_list = []
            for domain_info in domain_info_list:
                try:
                    merge_domain_list.extend(
                    [int(domain_info.split("-")[0]), int(domain_info.split("-")[1])])
                except ValueError:
                    print(f"Warning: 跳过无效 domain_info: '{domain_info}'")

            domain_start = min(merge_domain_list)
            domain_end = max(merge_domain_list)
            domain_info = "{0}-{1}".format(domain_start, domain_end)  # 更新连接后的domain信息
            domain_seq = self.protein_seq[domain_start - 1:domain_end]

            # 只统计长度占总长度10%以上的结构域(新算法不需要这么做)
            domain_length_percentage = (domain_end - domain_start) / len(self.protein_seq)
            if domain_length_percentage >= self.domain_length_threhold or True:
                    protein_name = "{0}({1}{2}-{3}{4} end)".format(self.gene_name, domain_seq[0], domain_start, domain_seq[-1], domain_end)

                    """
                    Biortus Blast获取DNA序列
                    """
                    # 生成序列文件
                    fasta_name = "{0}_{1}{2}-{3}{4}".format(self.gene_name, domain_seq[0], domain_start, domain_seq[-1], domain_end)
                    protein_seq_name = "{0}_{1}.fasta".format(self.uniprot_id, fasta_name)
                    protein_seq_path = os.path.join(self.work_dir, protein_seq_name)

                    with open(protein_seq_path, "w", encoding="utf-8") as f:
                        f.write(">{0}\n{1}".format(self.uniprot_id + "_" + protein_name, domain_seq))

                    # 上传至服务器, 生成blast结果
                    shared_blast_seq_path = os.path.join(shared_blast_seq_dir, protein_seq_name)
                    if os.path.exists(protein_seq_path):
                        shutil.copy(protein_seq_path, shared_blast_seq_path)

                        # 等待服务器回传Blast 结果
                        protein_xml_name = protein_seq_name.replace(".fasta", "_biortus.xml")
                        protein_xml_path = os.path.join(self.work_dir, protein_xml_name)
                        shared_blast_xml_path = os.path.join(shared_blast_xml_pir, protein_xml_name)
                        shared_blast_xml_path_dict[domain_seq] = {"shared": shared_blast_xml_path,
                                                                  "local": protein_xml_path,
                                                                  "domain_start": domain_start,
                                                                  "domain_end": domain_end
                                                                 }

        # 等待服务器回传Blast 结果, 获取模板DNA序列
        for domain_seq, xml_path_dict in shared_blast_xml_path_dict.items():

            protein_dna_seq = ""
            shared_blast_xml_path = xml_path_dict["shared"]
            protein_xml_path = xml_path_dict["local"]
            domain_start = xml_path_dict["domain_start"]
            domain_end = xml_path_dict["domain_end"]

            for times in range(30):

                if os.path.exists(shared_blast_xml_path):
                    shutil.copy(shared_blast_xml_path, protein_xml_path)
                    break

                time.sleep(10)

            # 解析xml
            if os.path.exists(protein_xml_path):
                with open(protein_xml_path, "r", encoding="utf-8") as f:
                    blast_xml_content = f.read()

                soup = BeautifulSoup(blast_xml_content, "xml")

                Hits_list = soup.find_all(name="Hit")

                PDB_write_status = 1
                for Hit_index, Hit in enumerate(Hits_list):
                    # get top 5 PDB
                    if PDB_write_status >= 2:
                        break

                    Hit_id = Hit.find(name="Hit_def").text
                    Truncation_length_begin = Hit.find(name="Hsp_query-from").text
                    Truncation_length_end = Hit.find(name="Hsp_query-to").text
                    Hit_identity = Hit.find(name="Hsp_identity").text
                    Hit_align_length = Hit.find(name="Hsp_align-len").text
                    hit_align_start = int(Hit.find(name="Hsp_hit-from").text)
                    hit_align_end = int(Hit.find(name="Hsp_hit-to").text)
                    identities = Truncation_length_begin + "-" + Truncation_length_end + " aa(" + str(
                        round(int(Hit_identity) / int(Hit_align_length) * 100, 2)) + "%)"
                    align_seq_length = int(Truncation_length_end) - int(Truncation_length_begin) + 1

                    # 判断匹配对象是否满足模板条件
                    if Hit_identity == Hit_align_length and align_seq_length == len(domain_seq):
                        # 导入质粒信息数据库
                        with open(plasmids_seq_db_path, "r", encoding="utf-8") as f:
                            plasmids_seq_db_dict = json.load(f)

                        if plasmids_seq_db_dict.get(Hit_id, None):
                            protein_dna_seq = plasmids_seq_db_dict[Hit_id]["dna seq"][(hit_align_start -1) * 3: hit_align_end * 3]

                    PDB_write_status += 1

            # protein_dna_seq = ""
            #
            # table = {
            #     'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
            #     'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
            #     'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
            #     'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
            #     'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
            #     'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
            #     'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
            #     'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
            #     'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
            #     'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
            #     'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
            #     'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
            #     'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
            #     'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
            #     'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*',
            #     'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W',
            # }
            #
            # for i in domain_seq:
            #     for key, value in table.items():
            #         if i == value:
            #             protein_dna_seq += key
            #             break

            # print(protein_dna_seq)
            # input("www:")
            for plasmid_verctor_name in self.plasmids_name_list:
                # 实例化质粒文档
                #tag_name = "8His-StrepII-8His-Sumo*"
                tag_name = "8His-StrepII-TEV-GG"
                protein_name = "{0}({1}{2}-{3}{4} end)".format(self.gene_name, domain_seq[0], domain_start,
                                                               domain_seq[-1], domain_end)
                save_dir = self.work_dir
                word_dir = self.word_dir

                plasmid_word_obj = GeneratePlasmidsWord(domain_seq, protein_dna_seq, plasmid_verctor_name, tag_name,protein_name, save_dir, word_dir)
                plasmid_word_obj.main()

                # print("插入Word是否能打开, 等待15s")
                # time.sleep(30)

    def insert_commerical_protein_info(self):
        # 导入PPT 模块
        ppt_app = Dispatch('PowerPoint.Application')
        ppt_template = ppt_app.Presentations.Open(self.pptx_save_path, WithWindow=False)

        # png初始插入位置
        png_left = 45
        png_top = 100
        png_width = 312
        png_height = 404

        # pdf初始插入位置
        pdf_left = 650
        pdf_top = 100
        pdf_width = 70
        pdf_height = 30

        # 初始插入PPT的位置
        insert_ppt_index = self.commercial_ppt_start_index
        # print("商品蛋白模板所在的初始PPT页数", self.commercial_ppt_start_index)

        # 商品蛋白信息所在的位置
        commercial_dir = os.path.join(self.work_dir, "commercial protein")

        for roots, dirs, files in os.walk(commercial_dir):
            for file in files:
                if re.search("\.pdf$", file) and not re.search("^~", file):
                    pdf_path = os.path.join(roots, file)
                    pdf_name = file.replace(".pdf", "")
                    png_path = os.path.join(roots, pdf_name, "page_1.png")

                    # print(insert_ppt_index)
                    ppt_template.Slides(insert_ppt_index).Shapes.AddOLEObject(pdf_left, pdf_top, pdf_width, pdf_height,
                                                                                           FileName=pdf_path,
                                                                                           IconLabel=pdf_name,
                                                                                           DisplayAsIcon=False)

                    ppt_template.Slides(insert_ppt_index).Shapes.AddPicture(FileName=png_path, LinkToFile=False,
                                                             SaveWithDocument=True, Left=png_left, Top=png_top,
                                                             Width=png_width, Height=png_height)

                    insert_ppt_index += 1

        # 保存目标PPT
        ppt_template.SaveAs(self.pptx_save_path)

        ppt_template.Close()
        ppt_app.Quit()

    def insert_plasmid_Word(self):

        # 插入表格对象
        # 导入PPT 模块
        ppt_app = Dispatch('PowerPoint.Application')
        ppt_template = ppt_app.Presentations.Open(self.pptx_save_path, WithWindow=False)
        print("ppt_template =", ppt_template, flush=True)
        print("ppt_template类型 =", type(ppt_template), flush=True)
        # print(ppt_template.Slides.Count)

        # 初始插入位置
        left = 15
        # left = 125
        top = 100
        # top = 50
        width = 70
        height = 70

        # 插入对象的个数
        insert_word_num = 0
        word_dir = os.path.join(self.work_dir, self.word_dir)

        for roots, dirs, files in os.walk(word_dir):
            for file in files:
                if re.search("\.docx$", file) and not re.search("^~", file):
                    word_path = os.path.join(roots, file)
                    word_name = file.replace(".docx", "")

                    ppt_template.Slides(ppt_template.Slides.Count - 2).Shapes.AddOLEObject(left, top, width, height,
                                                                               FileName=word_path,
                                                                               IconLabel=word_name, DisplayAsIcon=True)

                    # 每插入一个
                    # os.remove(word_path)
                    insert_word_num += 1
                    print("插入质粒Word文档 PPT, 已经插入{}个".format(insert_word_num), flush=True)

                    # 每插入一个, 左移125个单位
                    left += 125

                    # 每插入5个, 下移150个单位
                    if insert_word_num % 6 == 0:
                        top += 150
                        left = 15

        # 保存目标PPT
        ppt_template.SaveAs(self.pptx_save_path)

        ppt_template.Close()
        ppt_app.Quit()

    def iter_shape(self, shape):
        if type(shape) == pptx.shapes.group.GroupShape:
            for sshape in shape.shapes:
                if self.iter_shape(sshape):
                    return sshape
        else:
            if shape.has_text_frame:
                # text_shapes.append(shape)
                return shape

    def get_pdb_ligand_name(self, pdbid, entity_id):
        ligand_name = ""
        solvent_ions_list = ['FLC', 'MLT', 'BTB', 'PLP', 'CLT', 'FBP', '4NL', 'ANP', 'EPE',
                             'DTT', 'AMP', 'ADP', 'MES', 'CXS', 'NAI', 'NAD', 'NDP', 'NAP',
                             'AGS', 'BME', 'CAC', 'SO4', 'PO4', 'TRS', 'GOL', 'NAG', 'GTP',
                             'GNP', 'GDP', 'EDO', 'GOL', 'FAD', 'PLP', 'PEG', 'PG4', 'AGS',
                             'BCN', 'K', 'NA', 'MG', 'CA', 'LI', 'BE', 'SR', 'BA', 'F', 'CL',
                             'BR', 'I', 'FE', 'ZN', 'NI', 'MN', 'CR', 'CO', 'CU', 'AU', 'HG',
                             'CD', 'AL', 'PD', 'PT', 'DMS', 'MPD', 'CIT', 'TAM', 'PE8', 'EBE']#PDB的溶剂离子列表，不会写入ppt，而ppt中写入的是配体信息
        for times in range(10):
            try:
                response = requests.get(
                    url="https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{0}/{1}".format(pdbid, entity_id),
                    #proxies=proxies,
                    proxies=None,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
                        "Referer": "https://www.rcsb.org/",

                    }
                )
                if response.status_code == 200:
                    try:
                        ligand_name = response.json()["pdbx_entity_nonpoly"]["comp_id"]
                        if ligand_name in solvent_ions_list:
                            ligand_name = ""
                    except:
                        pass

                    break

            except:
                time.sleep(5)

        return ligand_name

    @staticmethod
    def check_broswer():
        # 尝试3次
        for i in range(3):
            broswer_version = Broswer.get_broswer_version(Params.broswer_name)

            if broswer_version:
                # 检查驱动是否存在
                chrome_driver_new_path = ProjectEvaluation.__chrome_driver_path.replace(".exe",
                                                                              "_V{0}.exe".format(broswer_version))

                if not os.path.exists(chrome_driver_new_path):
                    print("下载驱动中!", flush=True)

                    # 下载驱动
                    Broswer.download_chrome_driver(broswer_version, Params.chrome_driver_latest_version_path)

                    # 根据版本更改驱动名字
                    if os.path.exists(ProjectEvaluation.__chrome_driver_path):
                        os.rename(ProjectEvaluation.__chrome_driver_path, chrome_driver_new_path)
                        ProjectEvaluation.__chrome_driver_path = chrome_driver_new_path
                        Params.chrome_driver_path = chrome_driver_new_path
                        return True
                else:
                    # 驱动已经存在, 根据版本更改驱动名字
                    ProjectEvaluation.__chrome_driver_path = chrome_driver_new_path
                    Params.chrome_driver_path = chrome_driver_new_path
                    return True

            else:
                # 安装
                win32api.MessageBox(0, "Chrome浏览器未安装, 现在开始安装", "提醒", win32con.MB_TOPMOST)
                Broswer.install_chrome()
        else:
            # 安装浏览器或驱动失败
            return False

    @staticmethod
    def calculate_protein_property(protein_seq):

        # initial protein seq
        protein_seq = str(protein_seq).replace("*","")
        seq = ProteinAnalysis(protein_seq)

        # protein length
        length = str(len(protein_seq)) + " aa"

        # protein molecular weight
        if len(protein_seq) == 0:
            molecular_weight = "/"
        else:
            molecular_weight = round(seq.molecular_weight(),2)

        # protein isoelectric_point
        if len(protein_seq) == 0:
            isoelectric_point = "/"
        else:
            isoelectric_point = round(seq.isoelectric_point(),2)

        # mole number (1 microgram)
        if len(protein_seq) == 0:
            mole_num = "/"
        else:
            mole_num = str(round(1 / molecular_weight * 10 ** 6,2)) + " pMoles"

        # Molar Extinction coefficient
        if len(protein_seq) == 0:
            molar_extinction_coefficient_reduced = "/"
            molar_extinction_coefficient_disulfid = "/"
        else:
            molar_extinction_coefficient_reduced = seq.molar_extinction_coefficient()[0]
            molar_extinction_coefficient_disulfid = seq.molar_extinction_coefficient()[1]

        # A[280] of 1 mg/ml
        if len(protein_seq) == 0:
            absorb_reduced = "/"
            absorb_disulfid = "/"
        else:
            absorb_reduced = round(molar_extinction_coefficient_reduced / molecular_weight,2)
            absorb_disulfid = round(molar_extinction_coefficient_disulfid / molecular_weight,2)

        # 1 A[280] corr. to
        if len(protein_seq) == 0:
            reciprocal_absorb_reduced = "/"
            reciprocal_absorb_disulfid = "/"
        else:
            try:
                reciprocal_absorb_reduced = str(round(1 / absorb_reduced,2)) + " mg/ml"
                reciprocal_absorb_disulfid = str(round(1 / absorb_disulfid,2)) + " mg/ml"
            except:
                reciprocal_absorb_reduced = "/"
                reciprocal_absorb_disulfid = "/"

        # Charge at pH 7
        if len(protein_seq) == 0:
            charge_at_pH7 = "/"
        else:
            charge_at_pH7 = round(seq.charge_at_pH(7),2)

        # instability score
        protein_instability_score_weight_dict = {
            'W': {'W': 1.0, 'C': 1.0, 'M': 24.68, 'H': 24.68, 'Y': 1.0, 'F': 1.0, 'Q': 1.0, 'N': 13.34, 'I': 1.0, 'R': 1.0,
                  'D': 1.0, 'P': 1.0, 'T': -14.03, 'K': 1.0, 'E': 1.0, 'V': -7.49, 'S': 1.0, 'G': -9.37, 'A': -14.03,
                  'L': 13.34},
            'C': {'W': 24.68, 'C': 1.0, 'M': 33.6, 'H': 33.6, 'Y': 1.0, 'F': 1.0, 'Q': -6.54, 'N': 1.0, 'I': 1.0, 'R': 1.0,
                  'D': 20.26, 'P': 20.26, 'T': 33.6, 'K': 1.0, 'E': 1.0, 'V': -6.54, 'S': 1.0, 'G': 1.0, 'A': 1.0,
                  'L': 20.26},
            'M': {'W': 1.0, 'C': 1.0, 'M': -1.88, 'H': 58.28, 'Y': 24.68, 'F': 1.0, 'Q': -6.54, 'N': 1.0, 'I': 1.0,
                  'R': -6.54, 'D': 1.0, 'P': 44.94, 'T': -1.88, 'K': 1.0, 'E': 1.0, 'V': 1.0, 'S': 44.94, 'G': 1.0,
                  'A': 13.34, 'L': 1.0},
            'H': {'W': -1.88, 'C': 1.0, 'M': 1.0, 'H': 1.0, 'Y': 44.94, 'F': -9.37, 'Q': 1.0, 'N': 24.68, 'I': 44.94,
                  'R': 1.0, 'D': 1.0, 'P': -1.88, 'T': -6.54, 'K': 24.68, 'E': 1.0, 'V': 1.0, 'S': 1.0, 'G': -9.37,
                  'A': 1.0, 'L': 1.0},
            'Y': {'W': -9.37, 'C': 1.0, 'M': 44.94, 'H': 13.34, 'Y': 13.34, 'F': 1.0, 'Q': 1.0, 'N': 1.0, 'I': 1.0,
                  'R': -15.91, 'D': 24.68, 'P': 13.34, 'T': -7.49, 'K': 1.0, 'E': -6.54, 'V': 1.0, 'S': 1.0, 'G': -7.49,
                  'A': 24.68, 'L': 1.0},
            'F': {'W': 1.0, 'C': 1.0, 'M': 1.0, 'H': 1.0, 'Y': 33.6, 'F': 1.0, 'Q': 1.0, 'N': 1.0, 'I': 1.0, 'R': 1.0,
                  'D': 13.34, 'P': 20.26, 'T': 1.0, 'K': -14.03, 'E': 1.0, 'V': 1.0, 'S': 1.0, 'G': 1.0, 'A': 1.0,
                  'L': 1.0},
            'Q': {'W': 1.0, 'C': -6.54, 'M': 1.0, 'H': 1.0, 'Y': -6.54, 'F': -6.54, 'Q': 20.26, 'N': 1.0, 'I': 1.0,
                  'R': 1.0, 'D': 20.26, 'P': 20.26, 'T': 1.0, 'K': 1.0, 'E': 20.26, 'V': -6.54, 'S': 44.94, 'G': 1.0,
                  'A': 1.0, 'L': 1.0},
            'N': {'W': -9.37, 'C': -1.88, 'M': 1.0, 'H': 1.0, 'Y': 1.0, 'F': -14.03, 'Q': -6.54, 'N': 1.0, 'I': 44.94,
                  'R': 1.0, 'D': 1.0, 'P': -1.88, 'T': -7.49, 'K': 24.68, 'E': 1.0, 'V': 1.0, 'S': 1.0, 'G': -14.03,
                  'A': 1.0, 'L': 1.0},
            'I': {'W': 1.0, 'C': 1.0, 'M': 1.0, 'H': 13.34, 'Y': 1.0, 'F': 1.0, 'Q': 1.0, 'N': 1.0, 'I': 1.0, 'R': 1.0,
                  'D': 1.0, 'P': -1.88, 'T': 1.0, 'K': -7.49, 'E': 44.94, 'V': -7.49, 'S': 1.0, 'G': 1.0, 'A': 1.0,
                  'L': 20.26},
            'R': {'W': 58.28, 'C': 1.0, 'M': 1.0, 'H': 20.26, 'Y': -6.54, 'F': 1.0, 'Q': 20.26, 'N': 13.34, 'I': 1.0,
                  'R': 58.28, 'D': 1.0, 'P': 20.26, 'T': 1.0, 'K': 1.0, 'E': 1.0, 'V': 1.0, 'S': 44.94, 'G': -7.49,
                  'A': 1.0, 'L': 1.0},
            'D': {'W': 1.0, 'C': 1.0, 'M': 1.0, 'H': 1.0, 'Y': 1.0, 'F': -6.54, 'Q': 1.0, 'N': 1.0, 'I': 1.0, 'R': -6.54,
                  'D': 1.0, 'P': 1.0, 'T': -14.03, 'K': -7.49, 'E': 1.0, 'V': 1.0, 'S': 20.26, 'G': 1.0, 'A': 1.0,
                  'L': 1.0},
            'P': {'W': -1.88, 'C': -6.54, 'M': -6.54, 'H': 1.0, 'Y': 1.0, 'F': 20.26, 'Q': 20.26, 'N': 1.0, 'I': 1.0,
                  'R': -6.54, 'D': -6.54, 'P': 20.26, 'T': 1.0, 'K': 1.0, 'E': 18.38, 'V': 20.26, 'S': 20.26, 'G': 1.0,
                  'A': 20.26, 'L': 1.0},
            'T': {'W': -14.03, 'C': 1.0, 'M': 1.0, 'H': 1.0, 'Y': 1.0, 'F': 13.34, 'Q': -6.54, 'N': -14.03, 'I': 1.0,
                  'R': 1.0, 'D': 1.0, 'P': 1.0, 'T': 1.0, 'K': 1.0, 'E': 20.26, 'V': 1.0, 'S': 1.0, 'G': -7.49, 'A': 1.0,
                  'L': 1.0},
            'K': {'W': 1.0, 'C': 1.0, 'M': 33.6, 'H': 1.0, 'Y': 1.0, 'F': 1.0, 'Q': 24.68, 'N': 1.0, 'I': -7.49, 'R': 33.6,
                  'D': 1.0, 'P': -6.54, 'T': 1.0, 'K': 1.0, 'E': 1.0, 'V': -7.49, 'S': 1.0, 'G': -7.49, 'A': 1.0,
                  'L': -7.49},
            'E': {'W': -14.03, 'C': 44.94, 'M': 1.0, 'H': -6.54, 'Y': 1.0, 'F': 1.0, 'Q': 20.26, 'N': 1.0, 'I': 20.26,
                  'R': 1.0, 'D': 20.26, 'P': 20.26, 'T': 1.0, 'K': 1.0, 'E': 33.6, 'V': 1.0, 'S': 20.26, 'G': 1.0, 'A': 1.0,
                  'L': 1.0},
            'V': {'W': 1.0, 'C': 1.0, 'M': 1.0, 'H': 1.0, 'Y': -6.54, 'F': 1.0, 'Q': 1.0, 'N': 1.0, 'I': 1.0, 'R': 1.0,
                  'D': -14.03, 'P': 20.26, 'T': -7.49, 'K': -1.88, 'E': 1.0, 'V': 1.0, 'S': 1.0, 'G': -7.49, 'A': 1.0,
                  'L': 1.0},
            'S': {'W': 1.0, 'C': 33.6, 'M': 1.0, 'H': 1.0, 'Y': 1.0, 'F': 1.0, 'Q': 20.26, 'N': 1.0, 'I': 1.0, 'R': 20.26,
                  'D': 1.0, 'P': 44.94, 'T': 1.0, 'K': 1.0, 'E': 20.26, 'V': 1.0, 'S': 20.26, 'G': 1.0, 'A': 1.0, 'L': 1.0},
            'G': {'W': 13.34, 'C': 1.0, 'M': 1.0, 'H': 1.0, 'Y': -7.49, 'F': 1.0, 'Q': 1.0, 'N': -7.49, 'I': -7.49,
                  'R': 1.0, 'D': 1.0, 'P': 1.0, 'T': -7.49, 'K': -7.49, 'E': -6.54, 'V': 1.0, 'S': 1.0, 'G': 13.34,
                  'A': -7.49, 'L': 1.0},
            'A': {'W': 1.0, 'C': 44.94, 'M': 1.0, 'H': -7.49, 'Y': 1.0, 'F': 1.0, 'Q': 1.0, 'N': 1.0, 'I': 1.0, 'R': 1.0,
                  'D': -7.49, 'P': 20.26, 'T': 1.0, 'K': 1.0, 'E': 1.0, 'V': 1.0, 'S': 1.0, 'G': 1.0, 'A': 1.0, 'L': 1.0},
            'L': {'W': 24.68, 'C': 1.0, 'M': 1.0, 'H': 1.0, 'Y': 1.0, 'F': 1.0, 'Q': 33.6, 'N': 1.0, 'I': 1.0, 'R': 20.26,
                  'D': 1.0, 'P': 20.26, 'T': 1.0, 'K': -7.49, 'E': 1.0, 'V': 1.0, 'S': 1.0, 'G': 1.0, 'A': 1.0, 'L': 1.0}}

        instability_index = 0
        for i in range(0,len(protein_seq) - 1):
            instability_index += protein_instability_score_weight_dict[protein_seq[i]][protein_seq[i+1]]

        instability_index = round(instability_index * 10 / len(protein_seq),2)


        protein_property_dict = {"length":length,"molecular_weight":molecular_weight,"isoelectric_point":isoelectric_point,
                                 "mole_num":mole_num,"molar_extinction_coefficient_reduced":molar_extinction_coefficient_reduced,
                                 "absorb_reduced":absorb_reduced,"reciprocal_absorb_reduced":reciprocal_absorb_reduced,
                                "charge_at_pH7":charge_at_pH7,"instability_index":instability_index}

        return protein_property_dict

    @staticmethod
    def duplicate_slide(pres, index):
        template = pres.slides[index]
        try:
            blank_slide_layout = pres.slide_layouts[14]
        except:
            blank_slide_layout = pres.slide_layouts[len(pres.slide_layouts)]

        copied_slide = pres.slides.add_slide(blank_slide_layout)

        for shp in template.shapes:
            el = shp.element
            newel = copy.deepcopy(el)
            copied_slide.shapes._spTree.insert_element_before(newel, 'p:extLst')

        for _, value in six.iteritems(template.part.rels):
            # Make sure we don't copy a notesSlide relation as that won't exist
            if "notesSlide" not in value.reltype:
                copied_slide.part.rels.add_relationship(
                    value.reltype,
                    value._target,
                    value.rId
                )

        return copied_slide

    @staticmethod
    def del_slide(ppt_file,index):
        slides = list(ppt_file.slides._sldIdLst)
        ppt_file.slides._sldIdLst.remove(slides[index])

    @staticmethod
    def unzip_file(zip_src, dst_dir):
        r = zipfile.is_zipfile(zip_src)
        if r:
            fz = zipfile.ZipFile(zip_src, 'r')
            for file in fz.namelist():
                fz.extract(file, dst_dir)
        else:
            print('This is not zip', flush=True)

    def ppt_activity_of_the_protein(self, defalut_download_path="", waittime=60):
        file = self.ppt_file_handle

        # 判断是否已经下载过
        if not os.path.exists(os.path.join(self.work_dir,r'ppt_family_domain_modified.png')):

            # 抓取图片
            # 初始化浏览器
            bro = webdriver.Chrome(executable_path=self.chrome_driver_path)

            # 设置浏览器默认下载和阻止危害文件提醒弹出窗口
            options = webdriver.ChromeOptions()
            prefs = {'download.prompt_for_download': False, 'download.default_directory': defalut_download_path}
            options.add_experimental_option("prefs", prefs)
            bro = webdriver.Chrome(chrome_options=options,executable_path=self.chrome_driver_path)

            bro.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')
            params = {'cmd': 'Page.setDownloadBehavior',
                      'params': {'behavior': 'allow', 'downloadPath': defalut_download_path}}
            bro.execute("send_command", params)

            # 等待加载完成
            try:

                # 让浏览器对指定url发起访问,访问NCBI蛋白质比对网址
                bro.get("https://www.uniprot.org/uniprot/{}".format(self.uniprot_id))


                WebDriverWait(bro, waittime).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, '//*[@class="residue "]/img'))
                )

                bro.maximize_window()

                # 截图
                catalytic_activity_ele = bro.find_element_by_xpath('//div[@class="section "]//h4//*[@aria-expanded]')
                family_domain_ele = bro.find_element_by_xpath('//*[@id="family_and_domains"]/h2/span')

                js4 = "arguments[0].scrollIntoView();"

                bro.execute_script(js4, catalytic_activity_ele)

                bro.get_screenshot_as_file(os.path.join(self.work_dir,r'ppt_catalytic_activity.png'))

                bro.execute_script(js4, family_domain_ele)

                bro.get_screenshot_as_file(os.path.join(self.work_dir,r'ppt_family_domain.png'))

                bro.quit()

                im = Image.open(os.path.join(self.work_dir,r'ppt_catalytic_activity.png'))
                im = im.crop((im.size[0] * 0.15, 0, im.size[0], im.size[1] * 0.6))
                im.save(os.path.join(self.work_dir,r'ppt_catalytic_activity_modified.png'))


                im = Image.open(os.path.join(self.work_dir,r'ppt_family_domain.png'))
                im = im.crop((im.size[0] * 0.15, 0, im.size[0], im.size[1] * 0.6))
                im.save(os.path.join(self.work_dir,r'ppt_family_domain_modified.png'))

                # add slide 1, layout=2
                slice = file.slides.add_slide(file.slide_layouts[3])
                pptx_protein_title = slice.placeholders[18]
                title = pptx_protein_title.text_frame
                title_paragraph = title.add_paragraph()
                title_run = title.paragraphs[0].add_run()
                title_run.text = "Activity of the Protein"
                title_run.font.size = Pt(28)
                title_run.font.bold = True
                title_run.font.color.rgb = RGBColor(146, 50, 34)
                title_run.font.name = "Times New Roman"

                left = Cm(2.7)
                top = Cm(2.9)
                width = Cm(20)
                slice.shapes.add_picture(os.path.join(self.work_dir,r'ppt_catalytic_activity_modified.png'),left,top,width=width)

                left = Cm(0.2)
                top = Cm(9.73)
                width = Cm(25)
                slice.shapes.add_picture(os.path.join(self.work_dir,r'ppt_family_domain_modified.png'),left,top,width=width)

            except Exception as e:
                # print("Not Kinase ")
                bro.quit()

                # 添加Ppt
                slice = file.slides.add_slide(file.slide_layouts[3])
                pptx_protein_title = slice.placeholders[18]
                title = pptx_protein_title.text_frame
                title_paragraph = title.add_paragraph()
                title_run = title.paragraphs[0].add_run()
                title_run.text = "Activity of the Protein"
                title_run.font.size = Pt(28)
                title_run.font.bold = True
                title_run.font.color.rgb = RGBColor(146, 50, 34)
                title_run.font.name = "Times New Roman"


        else:

            im = Image.open(os.path.join(self.work_dir, r'ppt_catalytic_activity.png'))
            im = im.crop((im.size[0] * 0.15, 0, im.size[0], im.size[1] * 0.6))
            im.save(os.path.join(self.work_dir, r'ppt_catalytic_activity_modified.png'))

            im = Image.open(os.path.join(self.work_dir, r'ppt_family_domain.png'))
            im = im.crop((im.size[0] * 0.15, 0, im.size[0], im.size[1] * 0.6))
            im.save(os.path.join(self.work_dir, r'ppt_family_domain_modified.png'))

            # add slide 1, layout=2
            slice = file.slides.add_slide(file.slide_layouts[3])
            pptx_protein_title = slice.placeholders[18]
            title = pptx_protein_title.text_frame
            title_paragraph = title.add_paragraph()
            title_run = title.paragraphs[0].add_run()
            title_run.text = "Activity of the Protein"
            title_run.font.size = Pt(28)
            title_run.font.bold = True
            title_run.font.color.rgb = RGBColor(146, 50, 34)
            title_run.font.name = "Times New Roman"

            left = Cm(2.7)
            top = Cm(2.9)
            width = Cm(20)
            slice.shapes.add_picture(os.path.join(self.work_dir, r'ppt_catalytic_activity_modified.png'), left, top, width=width)

            left = Cm(0.2)
            top = Cm(9.73)
            width = Cm(25)
            slice.shapes.add_picture(os.path.join(self.work_dir, r'ppt_family_domain_modified.png'), left, top, width=width)

    def ppt_trans_membrane_helices_prediction(self, defalut_download_path="", waittime=60):
        file = self.ppt_file_handle

        # 判断是否下载过
        if not os.path.exists(os.path.join(self.work_dir, "DeepTMHMM-results", r'plot.png')):

            try:

                # # 初始化浏览器
                bro = webdriver.Chrome(executable_path=self.chrome_driver_path)

                # 设置浏览器默认下载和阻止危害文件提醒弹出窗口
                options = webdriver.ChromeOptions()
                prefs = {'download.prompt_for_download': False, 'download.default_directory': defalut_download_path}
                options.add_experimental_option("prefs", prefs)
                bro = webdriver.Chrome(chrome_options=options,executable_path=self.chrome_driver_path)

                bro.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')
                params = {'cmd': 'Page.setDownloadBehavior',
                          'params': {'behavior': 'allow', 'downloadPath': defalut_download_path}}
                bro.execute("send_command", params)

                # 让浏览器对指定url发起访问,访问NCBI蛋白质比对网址
                # bro.get("https://services.healthtech.dtu.dk/service.php?TMHMM-2.0")
                bro.get("https://dtu.biolib.com/DeepTMHMM")

                # # 接受协议
                # WebDriverWait(bro, waittime).until(
                #     EC.element_to_be_clickable(
                #         (By.XPATH, '//*[@id="cookiescript_accept"]'))
                # ).click()

                time.sleep(5)

                # # 等待序列输入框(下载按钮出现)
                # iframe = bro.find_element_by_xpath('//*[@id="servicetabs-1"]')
                # bro.switch_to_frame(iframe)

                WebDriverWait(bro, waittime).until(
                    EC.presence_of_element_located(
                        (By.XPATH, '//*[@id="app-section-content"]/div/div/div[2]/div[1]/div/div/form/div/div/div/div/div/div[1]/textarea'))
                ).send_keys(self.protein_seq)

                time.sleep(3)

                # 提交
                WebDriverWait(bro, waittime).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, '//*[@id="app-section-content"]/div/div/div[2]/div[1]/div/div/form/footer/div/span/span/a/span[1]/b'))
                ).click()

                # 等待结果加载，下载文件
                # WebDriverWait(bro, waittime).until(
                #     EC.element_to_be_clickable(
                #         (By.XPATH,
                #          '//*[@id="app-section-content"]/div/div/div[2]/div[1]/div/div/form/footer/div/span/span/a'))
                # )

                WebDriverWait(bro, waittime).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, '//*[@data-icon="download"]'))
                ).click()


                # 等待下载完成
                while True:
                    if os.path.exists(os.path.join(defalut_download_path, "DTU_DeepTMHMM_1.0.19-results.zip")):
                        break

                    time.sleep(5)


                # 解压文件
                self.unzip_file(os.path.join(defalut_download_path, "DTU_DeepTMHMM_1.0.19-results.zip"), os.path.join(defalut_download_path, "DeepTMHMM-results"))

                # bro.maximize_window()

                with open(os.path.join(self.work_dir, "DeepTMHMM-results", r'deeptmhmm_results.md'), "r", encoding="utf-8") as f:
                    for line in f:
                        if "Number of predicted TMRs:" in line:
                            # predict_result_ele = bro.find_element_by_xpath('/html/body/pre')
                            # num_of_predicted_TMHs = re.search("# WEBSEQUENCE Number of predicted TMHs:  (\d+)",predict_result_ele.text).group(1)
                            num_of_predicted_TMHs = line.split(":")[1].strip()
                            break
                    else:
                        num_of_predicted_TMHs = "0"

                # # 截图比对结果
                # trans_membrane_helices_prediction_ele = bro.find_element_by_xpath('/html/body/h2')
                #
                # js4 = "arguments[0].scrollIntoView();"
                #
                # bro.execute_script(js4, trans_membrane_helices_prediction_ele)
                #
                # bro.get_screenshot_as_file(os.path.join(work_dir,r'trans_membrane_helices_prediction.png'))

                bro.quit()

                # im = Image.open(os.path.join(work_dir,r'trans_membrane_helices_prediction.png'))
                # im = im.crop((0, 0, im.size[0] * 0.8, im.size[1]))
                # im.save(os.path.join(work_dir,r'trans_membrane_helices_prediction_modified.png'))

                # add slide 1, layout=2
                slice = file.slides.add_slide(file.slide_layouts[4])
                pptx_protein_title = slice.placeholders[18]
                title = pptx_protein_title.text_frame
                title_paragraph = title.add_paragraph()
                title_run = title.paragraphs[0].add_run()
                title_run.text = "Trans-membrane Helices Prediction"
                title_run.font.size = Pt(28)
                title_run.font.bold = True
                title_run.font.color.rgb = RGBColor(146, 50, 34)
                title_run.font.name = "Times New Roman"

                left = Cm(4.57)
                top = Cm(3.12)
                # height = Cm(13.15)
                pic = slice.shapes.add_picture(os.path.join(self.work_dir, "DeepTMHMM-results", r'plot.png'),left,top)

                # 图片置于文本框下方
                # slice.shapes._spTree.insert(1,pic._element)

                pptx_trans_membrane_helices_prediction_text = slice.placeholders[10]

                if num_of_predicted_TMHs == "0":
                    pptx_trans_membrane_helices_prediction_text.text = "Based on the result, this protein has none transmembrane, this protein should not be a membrane protein."
                else:
                    pptx_trans_membrane_helices_prediction_text.text = "Based on the result, this protein has {0} transmembrane.".format(num_of_predicted_TMHs)

                num_of_predicted_TMHs_dict = {"num_of_predicted_TMHs":num_of_predicted_TMHs}


                with open(os.path.join(self.work_dir,"num_of_predicted_TMHs_dict.json"),"w",encoding="utf-8") as f:
                    json.dump(num_of_predicted_TMHs_dict,f)

            except:
                # 出错

                slice = file.slides.add_slide(file.slide_layouts[4])
                pptx_protein_title = slice.placeholders[18]
                title = pptx_protein_title.text_frame
                title_paragraph = title.add_paragraph()
                title_run = title.paragraphs[0].add_run()
                title_run.text = "Trans-membrane Helices Prediction"
                title_run.font.size = Pt(28)
                title_run.font.bold = True
                title_run.font.color.rgb = RGBColor(146, 50, 34)
                title_run.font.name = "Times New Roman"



        else:
            # im = Image.open(os.path.join(work_dir,r'trans_membrane_helices_prediction.png'))
            # im = im.crop((0, 0, im.size[0] * 0.8, im.size[1]))
            # im.save(os.path.join(work_dir, r'trans_membrane_helices_prediction_modified.png'))

            # add slide 1, layout=2
            slice = file.slides.add_slide(file.slide_layouts[4])
            pptx_protein_title = slice.placeholders[18]
            title = pptx_protein_title.text_frame
            title_paragraph = title.add_paragraph()
            title_run = title.paragraphs[0].add_run()
            title_run.text = "Trans-membrane Helices Prediction"
            title_run.font.size = Pt(28)
            title_run.font.bold = True
            title_run.font.color.rgb = RGBColor(146, 50, 34)
            title_run.font.name = "Times New Roman"

            left = Cm(4.57)
            top = Cm(3.12)
            # height = Cm(13.15)
            pic = slice.shapes.add_picture(os.path.join(self.work_dir, "DeepTMHMM-results", r'plot.png'), left,
                                           top)

            # 图片置于文本框下方
            # slice.shapes._spTree.insert(1,pic._element)

            pptx_trans_membrane_helices_prediction_text = slice.placeholders[10]

            with open(os.path.join(self.work_dir, "num_of_predicted_TMHs_dict.json"), "r", encoding="utf-8") as f:
                num_of_predicted_TMHs_dict = json.load(f)

            num_of_predicted_TMHs = num_of_predicted_TMHs_dict["num_of_predicted_TMHs"]

            if num_of_predicted_TMHs == "0":
                pptx_trans_membrane_helices_prediction_text.text = "Based on the result, this protein has none transmembrane, this protein should not be a membrane protein."
            else:
                pptx_trans_membrane_helices_prediction_text.text = "Based on the result, this protein has {0} transmembrane.".format(
                    num_of_predicted_TMHs)

    def ppt_ss_and_pdb_structure_prediction(self, defalut_download_path="",waittime=60):
        file = self.ppt_file_handle

        # 判断是否提交过
        if not os.path.exists(os.path.join(self.work_dir,"ppt_ss_and_pdb_structure_prediction.txt")):

            try:

                # # 初始化浏览器
                bro = webdriver.Chrome(executable_path=self.chrome_driver_path)

                # 设置浏览器默认下载和阻止危害文件提醒弹出窗口
                options = webdriver.ChromeOptions()
                prefs = {'download.prompt_for_download': False, 'download.default_directory': defalut_download_path}
                options.add_experimental_option("prefs", prefs)
                bro = webdriver.Chrome(chrome_options=options,executable_path=self.chrome_driver_path)

                bro.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')
                params = {'cmd': 'Page.setDownloadBehavior',
                          'params': {'behavior': 'allow', 'downloadPath': defalut_download_path}}
                bro.execute("send_command", params)

                # 让浏览器对指定url发起访问,访问NCBI蛋白质比对网址
                bro.get("http://www.sbg.bio.ic.ac.uk/phyre2/html/page.cgi?id=index")

                # 等待序列输入框(下载按钮出现)
                WebDriverWait(bro, waittime).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, '/html/body/center[2]/form/table[1]/tbody/tr[3]/td[2]/textarea'))
                ).send_keys(self.protein_seq)

                # 输入结果接收邮箱
                WebDriverWait(bro, waittime).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, '/html/body/center[2]/form/table[1]/tbody/tr[1]/td[2]/input'))
                ).send_keys(self.email_address)

                # 输入任务名
                WebDriverWait(bro, waittime).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, '/html/body/center[2]/form/table[1]/tbody/tr[2]/td[2]/input'))
                ).send_keys(self.uniprot_id)

                # 提交
                WebDriverWait(bro, waittime).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, '/html/body/center[2]/form/table[1]/tbody/tr[7]/td[2]/input[1]'))
                ).click()


                bro.quit()

                # add slide 1, layout=2
                slice = file.slides.add_slide(file.slide_layouts[6])
                pptx_protein_title = slice.placeholders[18]
                title = pptx_protein_title.text_frame
                title_paragraph = title.add_paragraph()
                title_run = title.paragraphs[0].add_run()
                title_run.text = "Secondary Structure and Disorder Prediction"
                title_run.font.size = Pt(28)
                title_run.font.bold = True
                title_run.font.color.rgb = RGBColor(146, 50, 34)
                title_run.font.name = "Times New Roman"

                pptx_trans_membrane_helices_prediction_text = slice.placeholders[10]

                pptx_trans_membrane_helices_prediction_text.text = "There were several ****** regions in the sequences"

                # add slide 1, layout=2
                slice = file.slides.add_slide(file.slide_layouts[7])
                pptx_protein_title = slice.placeholders[18]
                title = pptx_protein_title.text_frame
                title_paragraph = title.add_paragraph()
                title_run = title.paragraphs[0].add_run()
                title_run.text = "PDB Structure"
                title_run.font.size = Pt(28)
                title_run.font.bold = True
                title_run.font.color.rgb = RGBColor(146, 50, 34)
                title_run.font.name = "Times New Roman"

                pptx_trans_membrane_helices_prediction_text = slice.placeholders[10]

                pptx_trans_membrane_helices_prediction_text.text = "The human homology has ****** determined"

                with open(os.path.join(self.work_dir,"ppt_ss_and_pdb_structure_prediction.txt"),"w",encoding="utf-8") as f:
                    f.write("success")

            except:
                # 出错
                slice = file.slides.add_slide(file.slide_layouts[6])
                pptx_protein_title = slice.placeholders[18]
                title = pptx_protein_title.text_frame
                title_paragraph = title.add_paragraph()
                title_run = title.paragraphs[0].add_run()
                title_run.text = "Secondary Structure and Disorder Prediction"
                title_run.font.size = Pt(28)
                title_run.font.bold = True
                title_run.font.color.rgb = RGBColor(146, 50, 34)
                title_run.font.name = "Times New Roman"


        else:

            # add slide 1, layout=2
            slice = file.slides.add_slide(file.slide_layouts[6])
            pptx_protein_title = slice.placeholders[18]
            title = pptx_protein_title.text_frame
            title_paragraph = title.add_paragraph()
            title_run = title.paragraphs[0].add_run()
            title_run.text = "Secondary Structure and Disorder Prediction"
            title_run.font.size = Pt(28)
            title_run.font.bold = True
            title_run.font.color.rgb = RGBColor(146, 50, 34)
            title_run.font.name = "Times New Roman"


            pptx_trans_membrane_helices_prediction_text = slice.placeholders[10]

            pptx_trans_membrane_helices_prediction_text.text = "There were several ****** regions in the sequences"

            # add slide 1, layout=2
            slice = file.slides.add_slide(file.slide_layouts[7])
            pptx_protein_title = slice.placeholders[18]
            title = pptx_protein_title.text_frame
            title_paragraph = title.add_paragraph()
            title_run = title.paragraphs[0].add_run()
            title_run.text = "PDB Structure"
            title_run.font.size = Pt(28)
            title_run.font.bold = True
            title_run.font.color.rgb = RGBColor(146, 50, 34)
            title_run.font.name = "Times New Roman"

            pptx_trans_membrane_helices_prediction_text = slice.placeholders[10]

            pptx_trans_membrane_helices_prediction_text.text = "The human homology has ****** determined"


@Gooey(program_name='项目评估',
       tabbed_groups=True,
       default_size=(800, 600),
       encoding="gbk",  # 设置编码格式，打包的时候遇到问题
       richtext_controls=True,  # 打开终端对颜色支持
       progress_regex=r"^progress: (\d+)%$",  # 正则，用于模式化运行时进度信息,
       navigation='Tabbed')
def main():

    # 可视化参数
    settings_msg = '项目评估'
    parser = GooeyParser(description=settings_msg)


    # 设置参数
    settings_group = parser.add_argument_group('Settings')

    # # 项目Uniprot ID
    # settings_group.add_argument("uniprotid", metavar='项目Uniprot ID',
    #                             help="项目Uniprot ID")
    #项目uniprot.txt文件
    #选择项是选择一个文件
    settings_group.add_argument("UniprotFile", metavar='项目Uniprot ID',help="项目Uniprot",widget='FileChooser',default=os.path.join(os.getcwd(),"uniprot.txt"))
    settings_group.add_argument("person_name", metavar='项目人名字',
                                help="项目人名字",default="张三(中文名字)")

    # # 预测二级结构时保留结果的个人邮箱
    # settings_group.add_argument("email_address", metavar='预测二级结构时保留结果的个人邮箱',
    #                             help="预测二级结构时保留结果的个人邮箱",default="***@wuxibiortus.com")

    # PPT 模板
    # settings_group.add_argument("pptx_template", metavar='PPT 模板',
    #                             help="PPT 模板",widget='FileChooser',default=os.path.join(os.getcwd(),"项目评估模板.pptx"))

    # Chrome 浏览器驱动程序
    settings_group.add_argument("chrome_driver", metavar='Chrome 浏览器驱动程序',
                                help="Chrome 浏览器驱动程序", widget='FileChooser', default=os.path.join(os.getcwd(), Params.chrome_driver_path))

    # 是否使用代理
    settings_group.add_argument("software_proxy_port", metavar='添加代理, 默认不使用代理',
                                help="输入代理端口", default="不使用代理")


    # 定义参数
    args = parser.parse_args()

    # project_evaluation_uniprot_id = "C6FG12"
    # email_address = "21feiyuner@gmail.com"
    # template_ppt_path = os.path.join(os.getcwd(),"项目评估模板.pptx")
    # chrome_driver_path = os.path.join(os.getcwd(),"chromedriver.exe")
    # person_name = "张三"

    project_evaluation_uniprot_all = args.UniprotFile
    person_name = args.person_name
    # email_address = args.email_address
    email_address = ""   # 占位用
    # template_ppt_path = args.pptx_template
    template_ppt_path = shared_template_ppt_path
    chrome_driver_path = args.chrome_driver
    software_proxy_port = args.software_proxy_port
    with open(project_evaluation_uniprot_all, "r", encoding="utf-8") as f:
        uniprot_list=f.readlines()
    for project_evaluation_uniprot_id in uniprot_list:

        match_status = True
        project_evaluation_uniprot_id = project_evaluation_uniprot_id.strip().replace('\n', '')
        # 获取这个文件夹下的所有pptx文件，如果project_evaluation_uniprot_id在pptx文件名中存在，则跳过
        pptx_files = os.listdir(r'D:\1项目评估自动化\ppt')
        for pptx_file in pptx_files:

            if pptx_file.endswith('pptx'):
                if project_evaluation_uniprot_id in pptx_file:
                    print('>>>>>>>已经存在：', project_evaluation_uniprot_id)
                    match_status = False
                    break
        if match_status:
            # try:
                print('>>>>>>>正在进行：', project_evaluation_uniprot_id)

                # 保存文件路径
                evaluation_file_path=r'D:\1项目评估自动化\项目评估信息'
                if not os.path.exists(evaluation_file_path):
                    os.makedirs(evaluation_file_path)
                work_dir = os.path.join(evaluation_file_path, project_evaluation_uniprot_id)

                # 添加代理
                if software_proxy_port.strip() == "不使用代理":
                    pass
                else:
                    Params.proxies_port = int(software_proxy_port)
                #逐行读取uniprot文件


                # 创建实例对象
                obj = ProjectEvaluation(project_evaluation_uniprot_id, person_name, email_address, template_ppt_path, chrome_driver_path, work_dir)

                # 创建实例对象(商品蛋白)
                commercial_protein_obj = GetProteinCommercialInfo(project_evaluation_uniprot_id, chrome_driver_path, work_dir)
                ##################
                # 运行程序(商品蛋白)
                ##################
                # 获取商品蛋白信息
                print("获取商品蛋白信息", flush=True)
                # commercial_protein_obj.main()

                ##################
                # 运行程序
                ##################
                # get_uniprot_html_content
                print("获取Uniprot网站信息", flush=True)
                obj.get_uniprot_html_content()

                # process_uniprot_html,return uniprot_content_dict
                obj.process_uniprot_html()
                # ppt cover
                obj.ppt_cover()

                # ppt protein sequence analysis
                print("分析序列信息", flush=True)
                obj.ppt_protein_sequence_analysis()
                # ppt protein infomation
                obj.ppt_protein_infomation()
                # ppt_activity_of_the_protein
                # ppt_activity_of_the_protein(file, project_evaluation_uniprot_id,chrome_driver_path,defalut_download_path = work_dir,waittime=60)

                # ppt_compounds_info
                obj.ppt_compounds_info(rows=7, columns=4)
                # ppt_trans_membrane_helices_prediction
                print("分析序列跨膜信息", flush=True)
                # ppt_trans_membrane_helices_prediction(file, protein_seq, chrome_driver_path, defalut_download_path=work_dir,
                #                                      waittime=300)

                # ppt_topology_prediction
                obj.ppt_topology_prediction(defalut_download_path=work_dir, waittime=120)

                # # ppt_ss_and_pdb_structure_prediction
                # print("分析序列二级结构信息", flush=True)
                # ppt_ss_and_pdb_structure_prediction(file, protein_seq, email_address, project_evaluation_uniprot_id,
                #                                     chrome_driver_path, defalut_download_path=work_dir, waittime=180)

                print("分析序列AF2结构信息", flush=True)
                # ppt_alphafold2_structure_prediction
                obj.ppt_alphafold2_structure_prediction()

                print("分析序列AF2结构PAE信息", flush=True)
                obj.ppt_alphafold2_PAE()

                # ppt_protein_domain_analysis
                print("分析序列Domain结构信息", flush=True)
                obj.ppt_protein_domain_analysis(defalut_download_path=work_dir, waittime=30)

                # uniprot pdb list
                print("分析序列PDB信息", flush=True)
                print("  -->uniprot PDB", flush=True)      
                obj.ppt_uniprot_pdb_summary()  

                # ppt_pdb_summary
                # get pdb id
                print("  -->pdb_summary", flush=True)
                for roots, dirs, files in os.walk(work_dir):
                    for xml_file in files:
                        if re.search("-Alignment\.xml$", xml_file):
                            obj.ppt_pdb_summary()

                # expression and purification summary
                obj.ppt_expression_and_purification_summary()

                # 保存ppt
                obj.pptx_save_name = "{0}_{1}_{2}_Biortus_Structure_Evaluation Report_{3}.pptx".format(obj.uniprot_id, time.strftime('%Y%m%d' , time.localtime()), obj.gene_name, obj.person_name).replace('/','_')

                obj.pptx_save_path = os.path.join(r'D:\1项目评估自动化\ppt', obj.pptx_save_name)
                obj.ppt_file_handle.save(obj.pptx_save_path)

                # ppt_reference
                # get pdb id
                print("分析序列Reference PDB信息", flush=True)
                for roots, dirs, files in os.walk(work_dir):
                    for xml_file in files:
                        if re.search("-Alignment\.xml$", xml_file):

                            obj.ppt_reference()

                # 重新复制ppt操作对象
                obj.ppt_file_handle = Presentation(obj.pptx_save_path)
                print("分析Biortus蛋白质粒信息", flush=True)
                obj.biortus_blast()

                ## 生成其他PPT
                # 复制母版(commercial and plasmid design)
                # read template ppt
                # file = Presentation(save_path)

                # 复制商品蛋白模板
                if commercial_protein_obj.pdfnum == 0:
                    commercial_ppt_template = 1
                else:
                    commercial_ppt_template = commercial_protein_obj.pdfnum

                for i in range(commercial_ppt_template):

                    copied_slide = obj.duplicate_slide(obj.ppt_file_handle, 0)

                # 复制质粒设计模板
                copied_slide = obj.duplicate_slide(obj.ppt_file_handle, 1)

                # 复制质粒设计word模板
                copied_slide = obj.duplicate_slide(obj.ppt_file_handle, 2)

                # 复制结束模板
                copied_slide = obj.duplicate_slide(obj.ppt_file_handle, 3)
                copied_slide = obj.duplicate_slide(obj.ppt_file_handle, 4)

                # 删除母版(commercial and plasmid design)
                obj.del_slide(obj.ppt_file_handle, 0)
                obj.del_slide(obj.ppt_file_handle, 0)
                obj.del_slide(obj.ppt_file_handle, 0)
                obj.del_slide(obj.ppt_file_handle, 0)
                obj.del_slide(obj.ppt_file_handle, 0)

                # 插入excel对象
                pass

                # 质粒设计
                print("生成质粒信息", flush=True)
                obj.plasmids_design()

                # 插入质粒Word文档
                print("生成质粒Word文档 PPT", flush=True)
                # obj.generate_plasmid_Word()
                obj.ppt_file_handle.save(obj.pptx_save_path)

                # 插入商品蛋白信息
                print("插入商品蛋白信息", flush=True)
                # obj.insert_commerical_protein_info()

                print("插入质粒Word文档 PPT", flush=True)
                # obj.insert_plasmid_Word()

                # 复制母版(end)
                # copied_slide = obj.duplicate_slide(obj.
                # ppt_file_handle, 0)
                # 删除母版(end)
                # obj.del_slide(obj.ppt_file_handle, 0)

                # 保存ppt
                # pptx_save_name = "{0}_{1}_{2}_WuxiBiortus_Structure_Evaluation Report_{3}.pptx".format(project_evaluation_uniprot_id,time.strftime('%Y%m%d' , time.localtime()), protein_gene_name, person_name)
                # save_path = os.path.join(os.getcwd(), pptx_save_name)
                # obj.ppt_file_handle.save(obj.pptx_save_path)
            # except  Exception as e:
            #     print(e)
            #     file_name = 'error.txt'
            #
            #     try:
            #         # 读取文件内容
            #         with open(file_name, 'r', encoding='utf-8') as f:
            #             existing_ids = set(line.strip() for line in f)  # 去掉换行符并存入集合
            #
            #         # 检查ID是否已存在
            #         if project_evaluation_uniprot_id not in existing_ids:
            #             # 如果不存在，追加写入
            #             with open(file_name, 'a', encoding='utf-8') as f:
            #                 f.write(project_evaluation_uniprot_id + '\n')
            #             print(f"{project_evaluation_uniprot_id} 已成功写入文件。")
            #         else:
            #             print(f"{project_evaluation_uniprot_id} 已存在，跳过写入。")
            #     except FileNotFoundError:
            #         # 如果文件不存在，则直接创建并写入
            #         with open(file_name, 'a', encoding='utf-8') as f:
            #             f.write(project_evaluation_uniprot_id + '\n')
            #         print(f"{file_name} 不存在，已创建并写入 {project_evaluation_uniprot_id}。")
            #     except Exception as e:
            #         print(f"发生错误：{e}")






if __name__ == '__main__':
    "参数设置"
    # 代理设置
    proxies = {
        "http": "http://127.0.0.1:10809",
        "https": "http://127.0.0.1:10809",
    }

    #表格格式设置
    # 字体
    # 表头字体
    title_font = Font(size=12, bold=True, name='Times New Roman')
    # 内容字体
    content_font = Font(size=10, name='Times New Roman')

    # 对齐
    # 姓名栏对齐
    name_column_align = Alignment(horizontal='left', vertical='center', wrap_text=True)

    # 内容对齐
    content_align = Alignment(horizontal='center', vertical='center', wrap_text=True)

    # 列的单位长度
    column_unit_length = 0.3

    # template ppt path
    # shared_template_ppt_path = r"\\192.168.1.52\佰翱得共享文件\生物部\生物公共资料\8. 计算结构平台\Project_Evaluation_Db\Template_PPT\项目评估模板.pptx"
    shared_template_ppt_path = r"\\10.50.34.10\佰翱得共享文件\生物部\生物公共资料\8. 计算结构平台\Project_Evaluation_Db\Template_PPT\项目评估模板20240910.pptx"

    # version file path
    shared_version_file_path = r"\\10.50.34.10\佰翱得共享文件\生物部\生物公共资料\8. 计算结构平台\Project_Evaluation_Db\Version\version.json"
    software_version = "20230921"

    # ppt db path
    # ppt_db_path = r"D:\Data\PyCharm\Data\项目部门\项目评估\PPT_DB"
    ppt_db_dict_path = r"\\10.50.34.10\佰翱得共享文件\生物部\生物公共资料\8. 计算结构平台\Project_Evaluation_Db\PPT_DB\pptx_pdb_info_dict.json"

    # plasmids_seq_db
    plasmids_seq_db_path = r"\\10.50.34.10\佰翱得共享文件\生物部\生物公共资料\8. 计算结构平台\Project_Evaluation_Db\Plamids_Seq_DB\plasimid_to_protein_dna_sequence.json"

    # Rcsb literature dir
    rcsb_literature_dir = r"\\10.50.34.10\佰翱得共享文件\生物部\生物公共资料\8. 计算结构平台\Project_Evaluation_Db\Rcsb_Literature"

    # chatpdf dir
    chatpdf_input_dir = r"\\10.50.34.10\佰翱得共享文件\生物部\生物公共资料\8. 计算结构平台\Project_Evaluation_Db\ChatPdf\Input"
    chatpdf_output_dir = r"\\10.50.34.10\佰翱得共享文件\生物部\生物公共资料\8. 计算结构平台\Project_Evaluation_Db\ChatPdf\Output"

    # shared server path
    shared_af2_pdb_dir = r"\\10.50.34.10\佰翱得共享文件\生物部\生物公共资料\8. 计算结构平台\Project_Evaluation_Db\AF2\pdb"
    shared_af2_png_pir = r"\\10.50.34.10\佰翱得共享文件\生物部\生物公共资料\8. 计算结构平台\Project_Evaluation_Db\AF2\png"

    shared_blast_seq_dir = r"\\10.50.34.10\佰翱得共享文件\生物部\生物公共资料\8. 计算结构平台\Project_Evaluation_Db\Seq_Blast\Seq"
    shared_blast_xml_pir = r"\\10.50.34.10\佰翱得共享文件\生物部\生物公共资料\8. 计算结构平台\Project_Evaluation_Db\Seq_Blast\Xml"

    shared_domain_parser_input_dir = r"\\10.50.34.10\佰翱得共享文件\生物部\生物公共资料\8. 计算结构平台\Project_Evaluation_Db\Domain_Parser\Input"
    shared_domain_parser_output_dir = r"\\10.50.34.10\佰翱得共享文件\生物部\生物公共资料\8. 计算结构平台\Project_Evaluation_Db\Domain_Parser\Output"

    if ProjectEvaluation.check_broswer():

        # 主程序
        main()

        print("结束", flush=True)

    else:

        win32api.MessageBox(0, "Chrome浏览器或驱动安装失败, 请手动安装重新运行程序, 或联系技术人员", "提醒", win32con.MB_TOPMOST)

    # 查看占位符的序号
    # for placeholder in slice.placeholders:
    #     info = placeholder.placeholder_format
    #     print("索引{0},名称{1},类型{2},文本{3}".format(info.idx,placeholder.name,info.type,placeholder.text))

    # 占位符添加图片
    # pptx_catalytic_activity_pic = slice.placeholders[10]
    # pptx_family_domain_pic = slice.placeholders[11]
    #
    # pptx_catalytic_activity_pic.insert_picture(r'ppt_catalytic_activity_modified.png')
    # pptx_family_domain_pic.insert_picture(r'ppt_family_domain_modified.png')

    # 修改图片尺寸
    # im = Image.open(r'topology_prediction.png')
    # im = im.crop((0, 0, im.size[0] * 0.5, im.size[1]))
    # im.save(r'trans_membrane_helices_prediction_modified.png')

查看代码，为什么我生成晶体的ppt很少，都是电镜的ppt

​‌
