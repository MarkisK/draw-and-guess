// Canvas init
var lc = LC.init(document.getElementById("lc"), {
    imageURLPrefix: '../static/_assets/lc-images',
    toolbarPosition: 'hidden',
    keyboardShortcuts: false,
    defaultStrokeWidth: 2,
    tools: [LC.tools.Pencil, LC.tools.Eraser]
});


// POST Drawing to Flask
$(function () {
    $('#submit').click(function () {  // Set to run when button with id 'guess' is clicked

        var image_data = JSON.stringify(lc.getSnapshot(['shapes', 'imageSize', 'position', 'scale']));  //gets image from canvas

        $.ajax({
            url: '/',
            data: image_data,
            type: 'POST',
            contentType: "application/json; charset=utf-8",
            success: function () {
                alert('POSTed')
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