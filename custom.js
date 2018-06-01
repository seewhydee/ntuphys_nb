code_show=true;
function code_toggle() {
    if (code_show) {
	$('div.input').hide();
    } else {
	$('div.input').show();
    }
    code_show = !code_show
}

define([
    'base/js/namespace',
    'base/js/promises'
], function(IPython, promises) {
    promises.app_initialized.then(function (appName) {
        if (appName !== 'NotebookApp') return;
        IPython.toolbar.add_buttons_group([
            {
                'label'   : 'Show/Hide Code',
                'icon'    : 'cogs',
                'callback': code_toggle
            }
        ]);
	code_toggle()
    });
});
