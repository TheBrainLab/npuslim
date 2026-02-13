document.addEventListener("DOMContentLoaded", function() {
    // 1. 找到那个语言切换按钮 (通过你之前定义的 id)
    var btn = document.getElementById("lang-switcher-btn");
    
    if (btn) {
        // 2. 获取当前页面的路径 (比如 /en/benchmark/index.html)
        var currentPath = window.location.pathname;
        var newPath = currentPath;

        // 3. 执行路径替换逻辑
        if (currentPath.includes("/zh_CN/")) {
            // 如果当前是中文，换成英文路径
            newPath = currentPath.replace("/zh_CN/", "/en/");
        } else if (currentPath.includes("/en/")) {
            // 如果当前是英文，换成中文路径
            newPath = currentPath.replace("/en/", "/zh_CN/");
        }
        
        // 4. 更新按钮的链接
        btn.setAttribute("href", newPath);

        // 5. 【补刀】强制在当前窗口打开 (解决新标签页问题)
        btn.removeAttribute("target");
        btn.setAttribute("target", "_self");
    }
});