function start_smart_process() {
    let progress = gradioApp().getElementById("sp_progress");
    let gallery = gradioApp().getElementById("sp_gallery");
    console.log("Requesting progress:", progress, gallery);
    requestProgress('sp', progress, gallery, atEnd, function(progress){});
    gradioApp().querySelector('#sp_error').innerHTML = '';
    return args_to_array(arguments);
}

function atEnd() {

}
