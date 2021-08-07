#!/usr/bin/env python
# coding: utf-8

import sys
import argparse
import typing
import notion.block
from itertools import takewhile
from notion.client import NotionClient


# https://raw.githubusercontent.com/echo724/notion2md/main/notion2md/exporter.py


class PageExporter:
    def __init__(self, page: notion.block.PageBlock):
        self._page = page

    def export_markdown(self, meta={}):
        md = "---\n"
        md += "title: %s\n" % self._page.title
        for k, v in meta.items():
            md += "%s: %s\n" % (k, v)
        md += "---\n\n"

        blocks = typing.cast(typing.List[notion.block.Block], list(self._page.children))
        md += self._blocks2md(blocks).strip()
        return md

    def _blocks2md(self, blocks: typing.List[notion.block.Block]):
        i = 0
        md = ""
        list_btypes = ["bulleted_list", "numbered_list", "to_do"]
        while i < len(blocks):
            block = blocks[i]
            if block.type in list_btypes:
                group = list(takewhile(lambda x: x.type == block.type, blocks[i:]))
                for block in group:
                    md += self._block2md(block)
                    md += "\n"
                md += "\n"
                i += len(group)
            else:
                md += self._block2md(block)
                md += "\n\n"
                i += 1
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
        elif btype == "table_of_contents":
            # 忽略 toc
            pass
        elif btype == "image":
            md += "![](%s)" % block.source
        elif block.children and btype != "page":
            for child in block.children:
                md += self._block2md(child, indent + 1)
        else:
            raise Exception("unsupport block type: %s" % btype)
        return md


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
    elements = block.get("properties", {}).get("title", [])
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


POST_URLS = {
    "2021-08-01-badger-txn": "https://www.notion.so/fleuria/badger-bdbd1620efd84038afedd9efc708ee66",
}


def main():
    parser = argparse.ArgumentParser(description="sync blog posts from notion")
    parser.add_argument(
        "--post", metavar="P", type=str, nargs=1, required=True, help="the key of post"
    )
    args = parser.parse_args()
    post_name = args.post[0]
    url = POST_URLS.get(post_name)
    if not url:
        print("%s not found" % post_name)
        sys.exit(1)
    download_post(post_name, url)


def download_post(post_name, url):
    token_v2 = open(".notion-token").read().strip()
    client = NotionClient(token_v2=token_v2)
    page = typing.cast(notion.block.PageBlock, client.get_block(url))
    exporter = PageExporter(page)
    md = exporter.export_markdown({"layout": "post"})
    file_name = "_posts/%s.md" % post_name
    with open(file_name, "w+") as f:
        f.write(md)


if __name__ == "__main__":
    main()
