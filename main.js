document.addEventListener('DOMContentLoaded', function() {
    document.body.addEventListener('htmx:beforeSwap', function(event) {
        $jsontree.destroyAll()
    });
    document.body.addEventListener('htmx:afterSwap', function(event) {
        console.log('Swapped element:', event.detail.target);

        window.setTimeout(function() {
            $jsontree.renderAll()
        }, 50);
    });

    // htmx.on("htmx:xhr:progress", function(evt, target, detail) {
    //     console.log("Event:", evt);
    //     console.log("Target:", target);
    //     console.log("Detail:", detail);
    //     console.log("Progress:", evt.detail.loaded, evt.detail.total);
    //     htmx.find("#progress").setAttribute("value", evt.detail.loaded/1000 * 100)
    // });

    document.body.addEventListener('htmx:sseMessage', function(event) {
        console.log("Event:", event);
        var data = JSON.parse(event.detail.message);
        if (data.progress !== undefined) {
            var progressBar = document.getElementById("progress-bar");
            var progressText = document.getElementById("progress-text");
            progressBar.value = data.progress;
            progressText.textContent = Math.round(data.progress) + "%";
        }
    });
});