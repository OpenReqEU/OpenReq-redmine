$(document).ready(function(){

    var $btn_clear = $("#" + ID_BTN_CLEAR);
    var $result = $("#" + ID_DIV_RESULT);

    //
    // Clear output
    $btn_clear.click(function(){
        // alert(post_url);
        $result.empty();
    });

    var $textArea = $("#" + ID_TEXTAREA);
    var $table_R = $('#' + ID_DATA_TABLE_R);
    var $table_S = $('#' + ID_DATA_TABLE_S);

    $table_R.on('click-row.bs.table', handle_tableOnClick);
    $table_S.on('click-row.bs.table', handle_tableOnClick);

    function handle_tableOnClick(e, row, $element) {
            // alert("HELP!");
            // $textArea.value.clear();
            $textArea.val(row["Value"]);
    }

});

function requestTableData_R(params) {
    requestTableData(params, "asr-review")
}

function requestTableData_S(params) {
    requestTableData(params, "asr-sentence")
}

function requestTableData(params, data_source) {
    // data you need
    console.log(params.data);

    // var options = $table.bootstrapTable('getOptions');

    $(document).ready(function () {
        $.ajax({
            url: URI_SERVICE_DATA_HANDLER + "/" + data_source,
            data: {
                "page" : params.data.offset / params.data.limit + 1,
                "per_page" : params.data.limit
            },
            type: 'GET',
            async: true,

            success: function(response) {
                //
                // add received data
                params.success({
                    total: response["total"],
                    rows: response["rows"]
                });
            },
            error: function(error) {
                console.log(error);
            }
        });

    });
}



