document.addEventListener("DOMContentLoaded", function() {
    // 1. Switch language button logic
    var btn = document.getElementById("lang-switcher-btn");
    
    if (btn) {
        // 2. Get the current path and determine the new path for switching languages
        var currentPath = window.location.pathname;
        var newPath = currentPath;

        // 3. Replace the language code in the path
        if (currentPath.includes("/zh_CN/")) {
            // If currently in Chinese, switch to English path
            newPath = currentPath.replace("/zh_CN/", "/en/");
        } else if (currentPath.includes("/en/")) {
            // If currently in English, switch to Chinese path
            newPath = currentPath.replace("/en/", "/zh_CN/");
        }
        
        // 4. Update the button's href to the new path
        btn.setAttribute("href", newPath);

        // 5. Ensure the link opens in the same tab
        btn.removeAttribute("target");
        btn.setAttribute("target", "_self");
    }
});