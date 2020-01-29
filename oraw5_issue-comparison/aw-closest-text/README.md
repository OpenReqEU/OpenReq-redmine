# aw-closest-text

This repository recommend related issues to checkout by calculating the jaccard distance beetween issues.

This service was created as a result of the OpenReq project funded by the European Union Horizon 2020 Research and Innovation programme under grant agreement No 732463.

## Build

```bash
docker build . -t aw-closest-text
```

## Run

```bash
docker run -d -p 8000:8000 aw-closest-text
```

## Usage

You can find the swagger docs [here](swagger.yaml).

### POST /getClosestTicket (JSON>JSON)

Parameters:

- content: `string` the text content of the issue
- ticketId: `number` the id of the issue
- treshold: `number` maximal distance of the issues being recommended beetween 0 and 1
- count: `number` the number of issue from the same project being returned
- project: `number`the project id

Returns:

- closest: `Object` the closest known issue
    - id `number`
    - distance `number`
- project_closests: `array[Object]` count closest issues withing the same project id
    - id `number`
    - distance `number`
