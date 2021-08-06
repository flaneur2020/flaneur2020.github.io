# https://raw.githubusercontent.com/echo724/notion2md/main/notion2md/exporter.py

import os
import requests
from datetime import datetime


class PageExporter:
    def __init__(self, url, client):
        self._client = client
        self._page = self._client.get_block(url)

    def export_markdown(self, image_dir):
        title = self._page.title
        pass

    def export_images(self, image_dir):
        pass

    def _md_page_header(self):
        """return the page's header formatted as Front Matter

          Returns:
            header(Stirng): return Front Matter header
        """
        header = "---\n"
        header += "title: {0}\n".format(self.title)
        try:
            header += "date: {0}\n".format(self._format_date())
        except:
            header += ""
        header += '---\n'
        return header

    def page2md(self, page=None):
        """change notion's block to markdown string
        """
        params = {'tap_count':0,'img_count':0,'num_index':0}
        md = ""
        for i, block in enumerate(page.children):
            try:
                md += self.block2md(block, params)
            except Exception as e:
                print(e)
        return md

    def block2md(self, block, params, indent=0):
        md = ""
        btype = block.type

        if btype == 'header':
            md += "# " + filter_inline_math(block)
        elif btype == "sub_header":
            md += "## " + filter_inline_math(block)
        elif btype == "sub_sub_header":
            md += "### " + filter_inline_math(block)
        elif btype == 'text':
            md += filter_inline_math(block)
        elif btype == 'bookmark':
            md += link_format(block.title, block.link)
        elif btype == "video" or btype == "file" or btype == "audio" or btype == "pdf" or btype == "gist":
            md += link_format(block.source, block.source)
        elif btype == "bulleted_list" or btype == "toggle":
            md += '- '+filter_inline_math(block)
        elif btype == "numbered_list":
            self.md += '1. '+filter_inline_math(block)
        elif btype == "code":
            md += "``` "+block.language.lower()+"\n"+block.title+"\n```"
        elif btype == "equation":
            md += "$$"+block.latex+"$$"
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
        elif block.children and btype != 'page':
            for child in block.children:
                md += self.block2md(child, params, indent+1)
        else:
            raise Exception("unsupport block type: %s" % btype)
        return md + "\n\n"


class PageBlockExporter:
    def __init__(self, url, client, blog_mode, main_dir="", image_dir="", download_dir=""):
        self.client = client
        self.page = self.client.get_block(url)
        self.title = self.page.title
        self.bmode = blog_mode
        if self.bmode:
            self.md = self._page_header()
            self.file_name = self._set_filename()
        else:
            self.file_name = self.page.title
            self.md = ""

        self.dir = main_dir

        self.image_dir = image_dir
        if self.image_dir is "":
            self.image_dir = os.path.join(self.dir, 'images/')

        self.download_dir = download_dir
        if self.download_dir is "":
            self.download_dir = os.path.join(self.dir, 'download/')
        self.sub_exporters = []

    def create_sub_folder(self):
        """create sub folder with current file name

          Args:
            directory(Stirng): set empty by default.
        """
        self.sub_dir = self.dir + 'subpage/'
        if not(os.path.isdir(self.sub_dir)):
            os.makedirs(os.path.join(self.sub_dir))

    def create_file(self):
        """create md file that md will be stored

          Returns:
            self.file(String): path of file
        """
        file_path = os.path.join(self.dir, self.file_name + '.md')
        self.file = open(file_path, 'w')
        return file_path

    def write_file(self):
        """save markdown output in the file
        """
        self.file.write(self.md)
        self.file.close()

    def ensure_image_foler(self):
        """create image output directory
        """
        if not(os.path.isdir(self.image_dir)):
            os.makedirs(os.path.join(self.image_dir))

    def image_export(self, url, count):
        """make image file based on url and count.

          Args:
            url(Stirng): url of image
            count(int): the number of image in the page

          Returns:
            image_path(String): image_path for the link in markdown
        """
        self.ensure_image_foler()
        image_path = self.image_dir + 'img_{0}.png'.format(count)
        r = requests.get(url, allow_redirects=True)
        open(image_path, 'wb').write(r.content)
        return image_path

    def ensure_download_foler(self):
        """create download output directory
        """
        print(self.download_dir)
        if not(os.path.isdir(self.download_dir)):
            os.makedirs(os.path.join(self.download_dir))

    def downlaod_file(self, url, file_name):
        """download a file in the page.

          Args:
            url(Stirng): url of the downlaod file
            file_name(String): name of the file

          Returns:
            None
        """
        self.ensure_download_foler()

        try:
            download_path = self.download_dir + file_name
        except Exception as e:
            print(e)
        r = requests.get(url, allow_redirects=True)
        open(download_path, 'wb').write(r.content)

    def _page_header(self):
        """return the page's header formatted as Front Matter

          Returns:
            header(Stirng): return Front Matter header
        """
        header = "---\n"
        header += "title: {0}\n".format(self.title)
        try:
            header += "date: {0}\n".format(self._format_date())
        except:
            header += ""
        tags = self._get_tags()
        if len(tags) != 0:
            header += "tags:\n"
            for tag in tags:
                header += '- ' + tag + '\n'
        header += '---\n'
        return header

    def _get_tags(self):
        """return tags in the page

          Condition:
            "Tags" or "tags" property should exit in the page

          Returns:
            tags([String]): tags in "Tags or tags" property in the page
        """
        try:
            tags = self.page.get_property('tags')
        except:
            print("\n[Notice] '{0}' has no Tags".format(self.page.title))
            tags = []
        return tags

    def _format_date(self):
        """return created date in the page

          Condition:
            "created" or "Created" property should exit in the page

          Returns:
            formatted_date(String): formatted created date
        """
        date = self.page.get_property("created_time")
        formatted_date = date.strftime('%Y-%m-%d')
        return formatted_date

    def _set_filename(self):
        """return formatted file name

          Returns:
            file name(String): formatted_file_name
        """
        try:
            date_in_name = self._format_date() + "-"
        except:
            print("[Notice] '{0}' has no Created Date".format(self.page.title))
            date_in_name = ""
        file_name = date_in_name + self.title.replace(" ", "-")
        return file_name

    def page2md(self, page=None):
        """change notion's block to markdown string
        """
        params = {'tap_count':0,'img_count':0,'num_index':0}
        if page is None:
            page = self.page
        for i,block in enumerate(page.children):
            try:
                self.block2md(block,params)
            except Exception as e:
                self.md += ""
        self.md = self.md[:-1]

    def block2md(self,block,params):
        if params['tap_count'] != 0:
            self.md += '\n'
            for i in range(params['tap_count']):
                self.md += '\t'
        try:
            btype = block.type
        except:
            pass
        if btype != "numbered_list":
            params['num_index'] = 0
        try:
            bt = block.title
        except:
            pass
        if btype == 'header':
            self.md += "# " + filter_inline_math(block)
        if btype == "sub_header":
            self.md += "## " + filter_inline_math(block)
        if btype == "sub_sub_header":
            self.md += "### " + filter_inline_math(block)
        if btype == 'page':
            self.create_sub_folder()
            sub_url = block.get_browseable_url()
            exporter = PageBlockExporter(sub_url, self.client, self.bmode)
            sub_page_path = exporter.create_file()
            try:
                if "https:" in block.icon:
                    icon = "!"+link_format("", block.icon)
                else:
                    icon = block.icon
            except:
                icon = ""
            self.sub_exporters.append(exporter)
            self.md += icon + link_format(exporter.file_name, sub_page_path)
        if btype == 'text':
            try:
                self.md += filter_inline_math(block)
            except:
                self.md += ""
        if btype == 'bookmark':
            self.md += link_format(bt, block.link)
        if btype == "video" or btype == "file" or btype == "audio" or btype == "pdf" or btype == "gist":
            self.md += link_format(block.source, block.source)
        if btype == "bulleted_list" or btype == "toggle":
            self.md += '- '+filter_inline_math(block)
        if btype == "numbered_list":
            params['num_index'] += 1
            self.md += str(params['num_index'])+'. '+filter_inline_math(block)
        if btype == "code":
            self.md += "``` "+block.language.lower()+"\n"+block.title+"\n```"
        if btype == "equation":
            self.md += "$$"+block.latex+"$$"
        if btype == "divider":
            self.md += "---"
        if btype == "to_do":
            if block.checked:
                self.md += "- [x] " + bt
            else:
                self.md += "- [ ]" + bt
        if btype == "quote":
            self.md += "> "+bt
        if btype == "column" or btype == "column_list":
            self.md += ""
        if btype == "collection_view":
            collection = block.collection
            self.md += self.make_table(collection)
        if block.children and btype != 'page':
            params['tap_count'] += 1
            for child in block.children:
                self.block2md(child,params)
            params['tap_count'] -= 1
        if params['tap_count'] == 0:
            self.md += "\n\n"

    def make_table(self, collection):
        columns = []
        row_blocks = collection.get_rows()
        for proptitle in row_blocks[0].schema:
            prop = proptitle['name']
            if prop == "Name":
                columns.insert(0, prop)
            else:
                columns.append(prop)
        table = []
        table.append(columns)
        for row in row_blocks:
            row_content = []
            for column in columns:
                if column == "Name" and row.get("content") is not None:
                    content = self.page2md(row)
                else:
                    content = row.get_property(column)
                if str(type(content)) == "<class 'list'>":
                    content = ', '.join(content)
                if str(type(content)) == "<class 'datetime.datetime'>":
                    content = content.strftime('%b %d, %Y')
                if column == "Name":
                    row_content.insert(0, content)
                else:
                    row_content.append(content)
            table.append(row_content)
        return table_to_markdown(table)


def link_format(name, url):
    """make markdown link format string
    """
    return "["+name+"]"+"("+url+")"


def table_to_markdown(table):
    md = ""
    md += join_with_vertical(table[0])
    md += "\n---|---|---\n"
    for row in table[1:]:
        if row != table[1]:
            md += '\n'
        md += join_with_vertical(row)
    return md


def join_with_vertical(list):
    return " | ".join(list)


def filter_inline_math(block):
    """This function will get inline math code and append it to the text
    """
    text = ""
    elements = block.get("properties")["title"]
    for i in elements:
        if i[0] == "⁍":
            text += "$$"+i[1][0][1]+"$$"
        else:
            text += block.title
    return text


def filter_source_url(block):
    try:
        return block.get('properties')['source'][0][0]
    except:
        return block.title
