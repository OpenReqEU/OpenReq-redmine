$(document).ready(function(){

    var $div_result = $("#" + ID_DIV_RESULT);
    var $textArea = $("#" + ID_TEXTAREA);
    var $text_form = $("#" + ID_TEXT_FORM);

    var $id_btn_rs = $("#" + ID_BTN_REMOVESTOPS);
    var $id_btn_rp = $("#" + ID_BTN_REMOVEPUNCTS);

    //
    // on click, POST the text to the preprocessor
    $id_btn_rs.click(function(){

        // disable the button
        // $(id_btn_rs).prop("disabled", true);

        $.ajax({
            url: URI_REMOVESTOPS,
            data: $text_form.serialize(),
            type: 'POST',
            async: true,

            success: function(response) {
                $div_result.append(response[RESPONSE_DATA_KEY] + "<br />");
                $id_btn_rs.prop("disabled", false);

            },
            error: function(error) {
                console.log(error);
                $id_btn_rs.prop("disabled", false);
            }
        });
    });

    $id_btn_rp.click(function(){

        // disable the button
        // $(id_btn_rp).prop("disabled", true);

        $.ajax({
            url: URI_REMOVEPUNCT,
            data: $text_form.serialize(),
            type: 'POST',
            async: true,

            success: function(response) {
                $div_result.append(response[RESPONSE_DATA_KEY] + "<br />");
                $id_btn_rp.prop("disabled", false);

            },
            error: function(error) {
                console.log(error);
                $id_btn_rp.prop("disabled", false);
            }
        });
    });

});