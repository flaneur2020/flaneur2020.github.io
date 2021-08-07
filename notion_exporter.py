# https://raw.githubusercontent.com/echo724/notion2md/main/notion2md/exporter.py

import os
import requests
from notion.client import NotionClient
from datetime import datetime


class PageExporter:
    def __init__(self, url: str, client: NotionClient):
        self._client = client
        self._page = self._client.get_block(url)

    def export_markdown(self, image_dir):
        return self._page2md(self._page)

    def export_images(self, image_dir):
        pass

    def _md_page_header(self, title, date):
        """return the page's header formatted as Front Matter

        Returns:
          header(Stirng): return Front Matter header
        """
        header = "---\n"
        header += "title: {0}\n".format(title)
        try:
            header += "date: {0}\n".format(date)
        except:
            header += ""
        header += "---\n"
        return header

    def _page2md(self, page):
        """change notion's block to markdown string"""
        md = ""
        for block in page.children:
            md += self._block2md(block)
        return md

    def _block2md(self, block, indent=0):
        md = ""
        btype = block.type
        if btype == "header":
            md += "# " + filter_inline_math(block)
        elif btype == "sub_header":
            md += "## " + filter_inline_math(block)
        elif btype == "sub_sub_header":
            md += "### " + filter_inline_math(block)
        elif btype == "text":
            md += filter_inline_math(block)
        elif btype == "bookmark":
            md += format_link(block.title, block.link)
        elif (
            btype == "video"
            or btype == "file"
            or btype == "audio"
            or btype == "pdf"
            or btype == "gist"
        ):
            md += format_link(block.source, block.source)
        elif btype == "bulleted_list" or btype == "toggle":
            md += "- " + filter_inline_math(block)
        elif btype == "numbered_list":
            md += "1. " + filter_inline_math(block)
        elif btype == "code":
            md += "``` " + block.language.lower() + "\n" + block.title + "\n```"
        elif btype == "equation":
            md += "$$" + block.latex + "$$"
        elif btype == "divider":
            md += "---"
        elif btype == "to_do":
            if block.checked:
                md += "- [x] " + block.title
            else:
                md += "- [ ]" + block.title
        elif btype == "quote":
            md += "> " + block.title
        elif btype == "column" or btype == "column_list":
            md += ""
        elif block.children and btype != "page":
            for child in block.children:
                md += self._block2md(child, indent + 1)
        else:
            raise Exception("unsupport block type: %s" % btype)
        return md + "\n\n"


def format_link(name, url):
    """make markdown link format string"""
    return "[" + name + "]" + "(" + url + ")"


def table_to_markdown(table):
    md = ""
    md += join_with_vertical(table[0])
    md += "\n---|---|---\n"
    for row in table[1:]:
        if row != table[1]:
            md += "\n"
        md += join_with_vertical(row)
    return md


def join_with_vertical(list):
    return " | ".join(list)


def filter_inline_math(block):
    """This function will get inline math code and append it to the text"""
    text = ""
    elements = block.get("properties")["title"]
    for i in elements:
        if i[0] == "⁍":
            text += "$$" + i[1][0][1] + "$$"
        else:
            text += block.title
    return text


def filter_source_url(block):
    try:
        return block.get("properties")["source"][0][0]
    except:
        return block.title


if __name__ == "__main__":
    token_v2 = open(".notion-token").read().strip()
    client = NotionClient(token_v2=token_v2)

    url = "https://www.notion.so/fleuria/badger-bdbd1620efd84038afedd9efc708ee66"
    exporter = PageExporter(url, client)
    md = exporter.export_markdown("./images/")
    print(md)
