$(document).ready(() => {
    $('.table td code').each( function () {
        text = $(this).text()
        text = text.replaceAll(/_([^\u200B])/g, '_\u200B$1').replaceAll(/([a-z])([A-Z])/g, '$1\u200B$2')
        $(this).text(text)
    })
})
