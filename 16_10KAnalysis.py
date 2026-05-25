# Databricks notebook source
#!pip install pdfplumber


# COMMAND ----------

import pdfplumber
import pypdfium2 as pdfium


# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

PDF_ROOT = "/Volumes/itesm/finanzas/nvda"

# COMMAND ----------

def list_pdfs():
    base = f"{PDF_ROOT}"
    return [f"{base}/{f}" for f in os.listdir(base) if f.endswith(".pdf")]

# COMMAND ----------

import os
list_pdfs()

# COMMAND ----------

def extract_page(pdf_path, page_i, ocr):
    text = pdfplumber.open(pdf_path).pages[page_i].extract_text() or ""
    if len(text.strip()) > 40:        # simple quality threshold
        return text, "digital"        # good digital text detected
    # fallback for scanned pages
    img = pdfium.PdfDocument(pdf_path)[page_i].render(scale=2).to_pil()
    return "\n".join(ocr.readtext(img, detail=0)), "ocr"

# COMMAND ----------

extract_page("/Volumes/itesm/finanzas/nvda/2022-10-27.pdf", 0, ocr)
import os
from concurrent.futures import ThreadPoolExecutor

def extract_pages(pdf_path, ocr):
    with ThreadPoolExecutor(max_workers=4) as executor:
        return [
            (page_i, *result)
            for page_i, result in enumerate(executor.map(lambda page_i: extract_page(pdf_path, page_i, ocr), range(len(pdfium.PdfDocument(pdf_path)))))
        ]
import os
from concurrent.futures import ThreadPoolExecutor



# COMMAND ----------

extract_page('/Volumes/itesm/finanzas/nvda/Receipt Date - 2023-Feb-24 - NVDA.OQ - NVIDIA Corp - 10-K - NVIDIA CORP 10-K - 23668751.pdf', 0, 'ocr')

# COMMAND ----------

extract_page('/Volumes/itesm/finanzas/nvda/Receipt Date - 2023-Feb-24 - NVDA.OQ - NVIDIA Corp - 10-K - NVIDIA CORP 10-K - 23668751.pdf', 1, 'ocr')

# COMMAND ----------

