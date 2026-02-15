import os
import sys

# -- Project information -----------------------------------------------------
project = "NPUSlim"
copyright = "2026, weiyangdaren"
author = "weiyangdaren"

if any("language=zh_CN" in arg for arg in sys.argv):
    language = "zh_CN"
    lang_btn_name = "English"
    lang_btn_url = "/en/index.html"
else:
    language = "en"
    lang_btn_name = "简体中文"
    lang_btn_url = "/zh_CN/index.html"

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

master_doc = "index"

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {"text": "NPUSlim"},
    # navbar configuration
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "search-field"],
    "navbar_persistent": ["theme-switcher"],
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
            # JS and CSS will look for this ID to attach the click event for language switching
            "attributes": {"target": "_self", "id": "lang-switcher-btn"},
        },
    ],
}

# -- Options for Edit this page -------------------------------------------------
html_context = {
    "github_user": "TheBrainLab",  # GitHub user
    "github_repo": "npuslim",  # Repository name
    "github_version": "main",  # Branch name
    "doc_path": "docs/source/",  # Documentation root path in the repository
}

# -- Options for internationalization -------------------------------------------------
locale_dirs = ["../locales/"]
gettext_compact = False
html_static_path = ["_static"]


def generate_root_files(app, exception):
    if exception is None and app.builder.name == "html":
        outdir = app.outdir
        if outdir.endswith("zh_CN") or outdir.endswith("en"):
            root_dir = os.path.abspath(os.path.join(outdir, ".."))
        else:
            root_dir = outdir

        redirect_html = """<!DOCTYPE html>
<html>
<head>
    <script type="text/javascript">
        var userLang = navigator.language || navigator.userLanguage; 
        if (userLang.startsWith('zh')) {
            window.location.replace("./zh_CN/index.html");
        } else {
            window.location.replace("./en/index.html");
        }
    </script>
    <meta http-equiv="refresh" content="0; url=./en/index.html">
    <title>Redirecting...</title>
</head>
<body>
    <p>Redirecting to documentation... <a href="./en/index.html">Click here</a></p>
</body>
</html>"""

        with open(os.path.join(root_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(redirect_html)

        open(os.path.join(root_dir, ".nojekyll"), "a").close()
        print(
            f"\n[NPUSlim] Auto-generated redirect index.html and .nojekyll in {root_dir}"
        )


# Register the JavaScript file for language switching
def setup(app):
    app.add_js_file("switch_lang.js")
    app.connect("build-finished", generate_root_files)
