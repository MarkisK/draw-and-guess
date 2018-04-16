// Canvas init
var lc = LC.init(document.getElementById("lc"), {
    imageURLPrefix: '../static/_assets/lc-images',
    toolbarPosition: 'hidden',
    keyboardShortcuts: false,
    defaultStrokeWidth: 1,
    tools: [LC.tools.Pencil, LC.tools.Eraser],
    imageSize: {width: 256, height: 256}
});

lc.setZoom(2);


// POST Drawing to Flask
$(function () {
    $('#submit').click(function (e) {  // Set to run when button with id 'guess' is clicked
        e.preventDefault();
        var image_export = lc.getImage().toDataURL();  // base64 PNG image

        $.ajax({
            url: '/',
            data: {
                base64: image_export
            },
            type: 'POST',
            success: function (response) {
                alert('Guess: ' + response.result)
            },
            error: function () {
                alert('error')
            }
        });
    });
});


// uncomment this to debug straight to page (not advised)
// function postImage() {
//     var obj = JSON.stringify(lc.getSnapshot(['shapes', 'imageSize', 'position', 'scale']))
//     document.getElementById("output").innerHTML = obj;
// }