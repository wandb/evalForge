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

    htmx.on('htmx:sseMessage', function(event) {
        console.log("Event:", event);
        
        var consoleOutput = document.getElementById("console_output");
        if (consoleOutput) {
            // Store the current scroll position and check if we're at the bottom
            var isScrolledToBottom = consoleOutput.scrollHeight - consoleOutput.clientHeight <= consoleOutput.scrollTop + 1;
            
            // Append the new message
            // (Assuming the message is being appended by htmx, we don't need to do it manually here)
            
            // If we were at the bottom before the new message, scroll to the bottom again
            if (isScrolledToBottom) {
                consoleOutput.scrollTop = consoleOutput.scrollHeight;
            }
        }
    });
});