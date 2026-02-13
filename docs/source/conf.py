import os
import sys

# -- Project information -----------------------------------------------------
project = 'NPUSlim'
copyright = '2026, weiyangdaren'
author = 'weiyangdaren'

# --- 1. 智能判断当前语言 (修复死循环) ---
# 检查编译命令里有没有传 '-D language=en'
# 如果有，说明现在正在编译英文版
if any("language=en" in arg for arg in sys.argv):
    language = 'en'
    lang_btn_name = "简体中文"
    # 英文版里，按钮应该跳回中文
    lang_btn_url = "/zh_CN/index.html"
else:
    language = 'zh_CN'  # 默认是中文
    lang_btn_name = "English"
    # 中文版里，按钮应该跳去英文
    lang_btn_url = "/en/index.html"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
    "dollarmath",
    "fieldlist",
]

master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {"text": "NPUSlim"},
    
    # 布局配置
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "search-field"],
    "navbar_persistent": ["theme-switcher"], # 只留日夜切换
    "navbar_align": "content", 
    "show_nav_level": 2,
    
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/weiyangdaren/npuslim",
            "icon": "fa-brands fa-github",
        },
        {
            "name": lang_btn_name,
            "url": lang_btn_url,
            "icon": "fa-solid fa-language",
            "type": "fontawesome",
            
            # --- 2. 这里的注释必须解开！！！ ---
            # 没有 id，JS 就找不到它；没有 target，就会新标签打开
            "attributes": {
                "target": "_self",       # 默认尝试当前窗口打开
                "id": "lang-switcher-btn" # 给 JS 留的暗号
            },
        },
    ],
}

# -- Options for Edit this page -------------------------------------------------

html_context = {
    "github_user": "TheBrainLab",       # GitHub 用户名
    "github_repo": "npuslim",           # 你的仓库名
    "github_version": "main",           # 分支名，通常是 main 或 master
    "doc_path": "docs/source/",         # 文档源码在仓库中的相对路径
}

# 翻译路径配置
locale_dirs = ['../locales/']
gettext_compact = False
html_static_path = ['_static']

# 注册 JS 脚本
def setup(app):
    app.add_js_file('switch_lang.js')