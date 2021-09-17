#!/usr/bin/env python
# coding: utf-8

"""sync-notion.py

Usage:
  sync-notion.py list
  sync-notion.py sync <POST>

Options:
  -h --help
"""

import sys
import docopt
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

        blocks = self._convert_children_to_blocks(self._page.children)
        md += self._blocks2md(blocks).strip()
        return md

    def _convert_children_to_blocks(
        self, children: notion.block.Children
    ) -> typing.List[notion.block.Block]:
        blocks = typing.cast(typing.List[notion.block.Block], list(children))
        return blocks

    def _flatten_blocks_with_children(
        self,
        blocks: typing.List[notion.block.Block],
        indent: int = 0,
    ) -> typing.List[typing.Tuple[notion.block.Block, int]]:
        result = []
        for block in blocks:
            result.append((block, indent))
            if block.children and len(block.children) > 0:
                children_blocks = self._convert_children_to_blocks(
                    block.children)
                result.extend(
                    self._flatten_blocks_with_children(
                        children_blocks, indent + 1)
                )
        return result

    def _blocks2md(self, blocks: typing.List[notion.block.Block], indent: int = 0):
        i = 0
        md = ""
        list_btypes = [
            notion.block.BulletedListBlock,
            notion.block.NumberedListBlock,
            notion.block.TodoBlock,
        ]
        block_with_indents = self._flatten_blocks_with_children(blocks)
        while i < len(block_with_indents):
            block, indent = block_with_indents[i]
            if type(block) in list_btypes:
                list_items = list(
                    takewhile(lambda x: type(x[0]) in list_btypes,
                              block_with_indents[i:])
                )
                for block, indent in list_items:
                    md += self._block2md(block, indent)
                    md += "\n"
                md += "\n"
                i += len(list_items)
            else:
                md += self._block2md(block, indent)
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
        else:
            raise Exception("unsupport block type: %s" % btype)

        indent_spaces = "    " * indent
        return indent_spaces + md


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
    "2021-08-14-rocksdb-txn": "https://www.notion.so/fleuria/rocksdb-a1bd4ae158be4b77b37e75bb210e105f",
    "2021-09-06-crdb-txn": "https://www.notion.so/fleuria/crdb-b54fc0cff91244a9acebd87d4ea6329c",
}


def main():
    opts = docopt.docopt(__doc__)
    if opts["list"]:
        items = POST_URLS.items()
        items = sorted(items, key=lambda x: x[0])
        for k, v in items:
            print("%s -- %s" % (k, v))
    elif opts["sync"]:
        post_name = opts["<POST>"]
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
