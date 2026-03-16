import os

if os.environ.get("NPUSLIM_PLUGIN_ENABLE", "0") == "1":
    try:
        import npuslim.plugins

        npuslim.plugins.register()
    except Exception as e:
        print(f"[NPUSlim] plugin registration failed: {e}")