$(document).ready(function(){

    //todo: check whether these two assignments can be uncommented
    var $div_result = $("#" + ID_DIV_RESULT);
    var $textArea = $("#" + ID_TEXTAREA);
    var $text_form = $("#" + ID_TEXT_FORM);

    var $btn_wordpos_tuples = $("#" + ID_BTN_WORDPOS_TUPLES);
    var $btn_postags_string = $("#" + ID_BTN_POSTAGS_STRING);
    var $btn_postag_phrases_string = $("#" + ID_BTN_POSTAGS_PHRASES_STRING);
    var $btn_postag_clauses_string = $("#" + ID_BTN_POSTAGS_CLAUSES_STRING);
    var $btn_postag_freq = $("#" + ID_BTN_POSTAG_FREQ);
    var $btn_sentence_count = $("#" + ID_BTN_SENTENCE_COUNT);

    //
    // on click, GET (word,pos) tuple list
    $btn_wordpos_tuples.click(function(){

        // disable the button
        // $(id_btn_rs).prop("disabled", true);

        $.ajax({
            url: URI_WORDPOS_TUPLES,
            data: $text_form.serialize(),
            type: 'POST',
            async: true,

            success: function(response) {
                $div_result.append(response[RESPONSE_DATA_KEY] + "<br />");
                $btn_wordpos_tuple.prop("disabled", false);

            },
            error: function(error) {
                console.log(error);
                $btn_wordpos_tuple.prop("disabled", false);
            }
        });
    });

    //
    // on click, POST text and receive POS string
    $btn_postags_string.click(function(){

        // disable the button
        // $(id_btn_rp).prop("disabled", true);

        $.ajax({
            url: URI_POSTAGS_STRING,
            data: $text_form.serialize(),
            type: 'POST',
            async: true,

            success: function(response) {
                $div_result.append(response[RESPONSE_DATA_KEY] + "<br />");
                $btn_postags_string.prop("disabled", false);

            },
            error: function(error) {
                console.log(error);
                $btn_postags_string.prop("disabled", false);
            }
        });
    });

    //
    // on click, POST text and receive POS string
    $btn_postag_freq.click(function(){

        // disable the button
        // $(id_btn_rp).prop("disabled", true);

        $.ajax({
            url: URI_POSTAG_FREQ,
            data: $text_form.serialize(),
            type: 'POST',
            async: true,

            success: function(response) {
                $div_result.append(response[RESPONSE_DATA_KEY] + "<br />");
                $btn_postag_freq.prop("disabled", false);

            },
            error: function(error) {
                console.log(error);
                $btn_postag_freq.prop("disabled", false);
            }
        });
    });

    //
    // on click, POST text and receive POS phrases string
    $btn_postag_clauses_string.click(function(){

        $.ajax({
            url: URI_POSTAGS_CLAUSES_STRING,
            data: $text_form.serialize(),
            type: 'POST',
            async: true,

            success: function(response) {
                $div_result.append(response[RESPONSE_DATA_KEY] + "<br />");
                $btn_postag_freq.prop("disabled", false);

            },
            error: function(error) {
                console.log(error);
                $btn_postag_freq.prop("disabled", false);
            }
        });
    });

    //
    // on click, POST text and receive POS phrases string
    $btn_postag_phrases_string.click(function(){

        $.ajax({
            url: URI_POSTAGS_PHRASES_STRING,
            data: $text_form.serialize(),
            type: 'POST',
            async: true,

            success: function(response) {
                $div_result.append(response[RESPONSE_DATA_KEY] + "<br />");
                $btn_postag_freq.prop("disabled", false);

            },
            error: function(error) {
                console.log(error);
                $btn_postag_freq.prop("disabled", false);
            }
        });
    });

    //
    // on click, POST text and receive sentence count
    $btn_sentence_count.click(function(){

        $.ajax({
            url: URI_SENTENCE_COUNT,
            data: $text_form.serialize(),
            type: 'POST',
            async: true,

            success: function(response) {
                $div_result.append(response[RESPONSE_DATA_KEY] + "<br />");
                $btn_postag_freq.prop("disabled", false);

            },
            error: function(error) {
                console.log(error);
                $btn_postag_freq.prop("disabled", false);
            }
        });
    });

});