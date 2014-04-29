'''
Created on 13.03.2014

@author: JoshLaptop
'''

from string import join
from uncertainties import UFloat
from string import Formatter
from preview import gui
from preview import conversion







class TableCell():
    def __init__(self, content=None):
        self.content = content
        self.fmtr = Formatter()
    def __repr__(self):
        return self.display()
    def display(self):
        """ type dependent string formatting """
        if isinstance(self.content, UFloat):
            return "{}".format(self.fmtr.format("{0:.1uS}", self.content))
        elif isinstance(self.content, int):
            return "{}".format(self.fmtr.format("{0:.0f}", self.content))
        elif isinstance(self.content, float):
            return "{}".format(self.fmtr.format("{0:.3f}", self.content))
        elif isinstance(self.content, basestring):
            return self.content
#        elif isinstance(self.content, unicode):
#            return self.content.encode(encoding='UTF-8')
        elif self.content is None:
            return "None"
        else:
            return str(self.content)


class TableElement(list):
    def __init__(self, *elements):
        if elements:
            self.element = []
            for elem in elements:
                if isinstance(elem, tuple) or isinstance(elem, list):
                    for subelem in elem:
                        self.element.append(TableCell(subelem))
                else:
                    cellcontent = TableCell(elem)
                    self.element.append(cellcontent)
        else:
            self.element = [TableCell()]

    def next(self):
         if not self.element:
             raise StopIteration
         return self.element.pop()

    def __getitem__(self, index):
        return self.element[index]

    def __setitem__(self, key, value):
        self.element[key] = TableCell(value)

    """multiplication with an integer changes
    the length of Element by this factor"""
    def __mul__(self, other):
        return self.__class__(self.element * other)

    def __repr__(self):
        return str(self.element)

    def append(self, other):
        self.element.append(other)

    def display(self):
        assert False, "the method display has to be defined by subclasses"


class TableRow(TableElement):

    def display(self):
        return(join(map(lambda x: str(x), self.element)))


class TableColumn(TableElement):
    def display(self):
        return(join(map(lambda x: str(x) + "\n", self.element), sep=""))


class Table:
    def __init__(self, position="!h", centered=True, siunitx=False):
        """
        Parameters
        ----------
        *position* :  "h"|"t"|"b" with or without leading "!",
        used to define the position of the table on the documentpage

        *centered* : True|False,
        used to define whether or not the table is centerd

        """
        self._title_rows = []
        self._title_columns = []
        self._rows = []
        self._columns = []
        self._row_count = 0
        self._column_count = 0
        self._row_seperator = False
        self._title_row_seperator = "single"
        self._border = False  # either True or False
        self._centered = centered  # either True or False
        self._position = position    # h,b,t with or without !
#        self.globalalign = ""
        self._column_separator = False
        self._column_alignment = []
        self._label_prefix = "tab:"
        self._label = ""
        self._caption = ""
        self._tablepath = ""
        self._siunitx = siunitx
# TODO: possibility to add colums and rows of differnet length

    def addColumn(self, column, align="c", title=None, symbol=None, unit=None):
        # processing of the column title row
        if not symbol is None:
            symbol = '$' + symbol + '$'
        if not unit is None:
            symbol = symbol + (" [\\si{{{}}}]" if self._siunitx
                               else " [{}]").format(unit)
        titlecolumn = [title, symbol]
        if not self._title_rows:
            for item in titlecolumn:
                if not item is None:
                    self._title_rows.append(TableRow(item))
        else:
            for (trow, item) in zip(self._title_rows, titlecolumn):
                if not item is None:
                    trow.append(TableCell(item))

        # processing of the table columns
        self._column_alignment.append(align)
        self._columns.append(TableColumn(column))
        self._column_count += 1

        # processing of the table rows
        if not self._rows:
            for item in column:
                self._rows.append(TableRow(item))
            self._row_count += len(column)
        else:
            for (row, item) in zip(self._rows, column):
                row.append(TableCell(item))

    def addRow(self, row):
        self._rows.append(TableRow(row))
        self._row_count += 1
        if not self._columns:
            for item in row:
                self._columns.append(TableColumn(item))
            self._column_count += len(row)
        else:
            for (col, item) in zip(self._columns, row):
                col.append(TableCell(item))

    def layout(self, seperator="none",
               title_row_seperator="single", border=False):
        """
        Method to set the grid and border layout of the table

        Parameters
        ----------
        *seperator* = "none"|"row"|"column"|"both",
        used to set the gridstyle of the table

        *title_row_seperator* = "single"|"double"|"none",
        used to set the style of the seperatorline beneath the title row

        *border* = False|True,
        used to en- or disable the border of the table
        """

        if seperator == "none":
            self._row_seperator = self._column_separator = False
        elif seperator == "both":
            self._row_seperator = self._column_separator = True
        elif seperator == "row":
            self._row_seperator = True
            self._column_separator = False
        elif seperator == "column":
            self._row_seperator = False
            self._column_separator = True
        self._title_row_seperator = title_row_seperator
        self._border = border

    def caption(self, caption):
        self._caption = caption

    def label(self, label, label_prefix="tab:"):
        self._label = label
        self._label_prefix = label_prefix



    def _latexColumns(self):
        column_settings = "|{}|" if self._border else "{}"
        seperator = "|" if self._column_separator else ""
        column_settings = column_settings.format(join(self._column_alignment,
                                                    sep=seperator))
        self._column_settings = column_settings
        return column_settings

    def _rowEnd(self):
        return "\\\\\hline" if self._row_seperator else "\\\\"

    def _latexTitleRows(self):
        trows = []
        if self._title_rows:
            for trow in self._title_rows:
                trows.append(join(map(lambda s: str(s), trow.element),
                             sep=" & ") + "\\\\" + "\n")
        return trows

    def _latexRows(self):
        rows = []
        for row in self._rows:
            if self._siunitx:
                rows.append(join(map(lambda s: "\\num{{{}}}".format(str(s)),
                                     row.element), sep=" & ")
                            + self._rowEnd() + "\n")
            else:
                rows.append(join(map(lambda s: str(s), row.element), sep=" & ")
                            + self._rowEnd() + "\n")

        if self._row_seperator:
            rows[-1] = rows[-1].replace("\\hline", "")
        return rows

    def _latexTable(self, path):
        tablebegin = "\\begin{table}" + "[{}]\n".format(self._position)
        tableend = "\\end{table}\n"
        tabularbegin = "\t\\begin{tabular}" + \
                       "{{{}}}\n".format(self._latexColumns())
        tabularend = "\t\\end{tabular}\n"
        label = "\\label{{{}}}".format(self._label_prefix + self._label)
        caption = "\t\\caption{{{} {}}}\n".format(self._caption, label)
#        print(tablebegin, tableend, tabularbegin, tabularend, label, caption)
        texfile = open(path, "w")
        texfile.write(tablebegin +
                      ("\t\\centering\n" if self._centered else "")
                      + tabularbegin)
        if self._border:
            texfile.write("\t\t\\hline\n")
        if self._title_rows:
            for line in self._latexTitleRows():
                texfile.write("\t\t" + line)
            if self._title_row_seperator == "single":
                texfile.write("\\hline")
                texfile.write("\n")
            elif self._title_row_seperator == "double":
                texfile.write("\\hline")
                texfile.write("\\hline")
                texfile.write("\n")

        for line in self._latexRows():
            texfile.write("\t\t" + line)
        if self._border:
            texfile.write("\t\t\\hline\n")
        texfile.write(tabularend + caption + tableend)
        texfile.close()

    def save(self, savepath):
        """
        Saves the table ".tex" file in the
        designated *savepath*

        """
        self._tablepath = savepath
        self._latexTable(self._tablepath)
        self._latexTable(savepath)

    def _tempsave(self, savepath):
        self._latexTable(savepath)

    def show(self, quiet=True):
        """
        Opens a GUI displaying a ".png" file representation of the table.
        PyQt4 is needed to use this method.
        """
        if not self._tablepath:
            conv = conversion.Converter(tableobj=self, quiet=quiet)
        else:
            conv = conversion.Converter(self._tablepath, quiet=quiet)
        gui._show(conv.texToPng(), conv)


    def __repr__(self):
        return join(map(lambda row: row.display(), self._rows), sep="\n")
