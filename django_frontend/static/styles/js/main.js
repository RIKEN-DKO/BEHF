$(window).on('load', function () {
    $(document).ready(function () {
        console.log('Document is ready');
        console.log($(".details summary").length + ' summary elements found');

        $(".details summary").click(function () {
            console.log('Click');
            var docId = $(this).data('doc-id').toString();  // convert docId to string
            var detailsElement = $(this).parent();
            var textAeEditor = detailsElement.find('.textae-editor');

            $.ajax({
                url: 'http://localhost:5555',  // modify this as per your setup
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    query: docId,
                    num_res: null,
                    type: "get_doc_annotations"
                }),
                success: function (data) {
                    textAeEditor.html(JSON.stringify(data));
                    // TODO: Use TextAE library to render the annotations
                },
                error: function (error) {
                    console.error(error);
                    textAeEditor.html('Failed to load annotations.');
                }
            });
        });
    });
});
