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
});