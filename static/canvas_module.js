

var lc = LC.init(document.getElementById("lc"), {
    imageURLPrefix: '../static/_assets/lc-images',
    toolbarPosition: 'hidden',
    keyboardShortcuts: false,
    defaultStrokeWidth: 2,
    tools: [LC.tools.Pencil, LC.tools.Eraser]
});

function jsonify() {
    var obj = JSON.stringify(lc.getSnapshot(['shapes', 'imageSize', 'position', 'scale']))
    document.getElementById("output").innerHTML = obj;
}
