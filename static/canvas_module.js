var lc = LC.init(document.getElementById("lc"), {
    imageURLPrefix: '../static/_assets/lc-images',
    toolbarPosition: 'bottom',
    defaultStrokeWidth: 3,
    strokeWidths: [1, 2, 3, 5, 30]
});

function jsonify() {
    var obj = JSON.stringify(lc.getSnapshot(['shapes', 'imageSize', 'position', 'scale']))
    document.getElementById("output").innerHTML = obj;
}
