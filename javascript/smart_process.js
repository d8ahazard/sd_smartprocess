function start_smart_process() {
    requestProgress('sp');
    gradioApp().querySelector('#sp_error').innerHTML = '';
    return args_to_array(arguments);
}

onUiUpdate(function () {
    check_progressbar('sp', 'sp_progressbar', 'sp_progress_span', '', 'sp_interrupt', 'sp_preview', 'sp_gallery')
})