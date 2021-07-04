const address = "http://127.0.0.1:5000"
//const address = "http://mmc-cuda04.informatik.uni-augsburg.de:5000"
//const address = "http://gimpel.informatik.uni-augsburg.de:5000"

function updateNetworkCheckpoint(text_element) {
    $.ajax({
        url: address + "/get_network_checkpoint",
        type: "GET",
        contentType: 'application/json',
        error: function (textStatus, errorThrown) {
            console.log("status", textStatus)
            console.log("errorThrown", errorThrown)
        },
        success: function (data) {
            text_element.text('current network checkpoint: ' + data['network_checkpoint'])
        }
    })
}

function buildCarTable(table, id, with_rotation) {
    $.ajax({
        url: address + "/get_classes",
        type: "GET",
        contentType: 'application/json',
        error: function (textStatus, errorThrown) {
            console.log("status", textStatus)
            console.log("errorThrown", errorThrown)
        },
        success: function (data) {
            const labels = data['labels']
            const random_label = data['random_label']
            var rows = []
            for (var i = 0; i < labels.length; i++) {
                const label_name = labels[i]['name']
                if (label_name === 'Rotation' && !with_rotation) {
                    continue
                }
                const label_classes = labels[i]['classes']

                rows.push('<tr><td>' + label_name)
                rows.push('<td><form><label><select id="' + label_name + '_' + id + '">')
                rows.push('<option value="-1">any</option>')
                for (var j = 0; j < label_classes.length; j++) {
                    if (j === random_label[label_name]) {
                        rows.push('<option value="' + j + '" selected="selected">' + label_classes[j] + '</option>')
                    } else {
                        rows.push('<option value="' + j + '">' + label_classes[j] + '</option>')
                    }
                }
                rows.push('</select></label></form></td></tr>')
            }
            rows.push('<tr><td>Seed</td><td><form><label for="seed"></label><input class="attribute" type="text" id="Seed_' + id + '" name="seed" value="0"></form></td></tr>')
            rows.push('<tr><td>Randomize Seed</td><td><form><label for="randomize_seed"></label>')
            rows.push('<input type="checkbox" id="Randomize_seed_' + id + '" class="randomize_seed" name="randomize_seed" checked>')
            rows.push('</form></td></tr>')

            table.append(rows.join(''));
        }
    })
}

function getAttributesFromTable(id, size) {
    manufacturer = document.getElementById('Manufacturer_' + id).value
    model = document.getElementById('Model_' + id).value
    color = document.getElementById('Color_' + id).value
    body = document.getElementById('Body_' + id).value
    rotation = ''
    if (document.getElementById('Rotation_' + id) != null) {
        rotation = document.getElementById('Rotation_' + id).value
    }
    ratio = document.getElementById('Ratio_' + id).value
    background = document.getElementById('Background_' + id).value
    seed = document.getElementById('Seed_' + id).value
    randomize_seed = $('#Randomize_seed_' + id).is(':checked')

    result = {
        "manufacturer": manufacturer,
        "model": model,
        "color": color,
        "body": body,
        "rotation": rotation,
        "ratio": ratio,
        "background": background,
        "seed": seed,
        "randomize_seed": randomize_seed,
        "size": size
    }
    return result
}

