const SP_PROGRESSBAR_LABEL = 'sp_preview'
const SP_GALLERY_LABEL = 'sp_gallery'
const SP_ERROR_LABEL = '#sp_error'
const SP_GALLERY_CHILD = 'sp_gallery_kid';
const SP_PROGRESS_LABEL = 'sp_progress';

function start_smart_process() {
    rememberGallerySelection(SP_GALLERY_LABEL)
    gradioApp().querySelector('#sp_error').innerHTML = ''
    var spGalleryElt = gradioApp().getElementById(SP_GALLERY_LABEL)
    // set id of first child of spGalleryElt to 'sp_gallery_kid',
    // required by AUTOMATIC1111 UI Logic
    spGalleryElt.children[0].id = SP_GALLERY_CHILD
    var id = randomId();
    requestProgress(id,
        gradioApp().getElementById(SP_GALLERY_LABEL),
        gradioApp().getElementById(SP_GALLERY_CHILD),
        function () {
        },
        function (progress) {
            gradioApp().getElementById(SP_PROGRESS_LABEL).innerHTML = progress.textinfo
        })

    const argsToArray = args_to_array(arguments);
    argsToArray.push(argsToArray[0])
    argsToArray[0] = id;
    return argsToArray
}

onUiUpdate(function(){
    check_gallery(SP_GALLERY_LABEL)
})
