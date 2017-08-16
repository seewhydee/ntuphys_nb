code_show=true;
function code_toggle() {
    if (code_show) {
	$('div.input').hide();
    } else {
	$('div.input').show();
    }
    code_show = !code_show
}

$([IPython.events]).on('notebook_loaded.Notebook', function() {
    $("#view_menu").append("<li id=\"toggle_input\" title=\"Show/Hide Code\"><a href=\"javascript:code_toggle()\">Show/Hide Code</a></li>")
    $('div#ipython_notebook').hide()
    $('span#save_widget').hide()
    $('span#kernel_logo_widget').hide()
    // $('div.input').hide()
    // $('#notebook_panel').append(copyright)
});
