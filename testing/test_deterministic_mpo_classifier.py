"""
def test_add_mpos():

    # Check mpos are added in the right place (check for 3, for speed)
    added_mpos = mpo_train[0]
    for mpo in mpo_train[1:3]:
        added_mpos = add_mpos(added_mpos, mpo)

    # Check data of added MPOs (checks shape too).
    for i, (site0, site1, site2, site3) in enumerate(
        zip(
            mpo_train[0].tensors,
            mpo_train[1].tensors,
            mpo_train[2].tensors,
            added_mpos.tensors,
        )
    ):

        if i == 0:
            assert np.array_equal(site0.data, site3.data[:, :, :, : site0.shape[3]])
            assert np.array_equal(
                site1.data,
                site3.data[:, :, :, site0.shape[3] : site0.shape[3] + site1.shape[3]],
            )
            assert np.array_equal(
                site2.data,
                site3.data[
                    :,
                    :,
                    :,
                    site0.shape[3]
                    + site1.shape[3] : site0.shape[3]
                    + site1.shape[3]
                    + site2.shape[3],
                ],
            )

        elif i == (mpo_train[0].num_tensors - 1):
            assert np.array_equal(site0.data, site3.data[:, :, : site0.shape[2], :])
            assert np.array_equal(
                site1.data,
                site3.data[:, :, site0.shape[2] : site0.shape[2] + site1.shape[2], :],
            )
            assert np.array_equal(
                site2.data,
                site3.data[
                    :,
                    :,
                    site0.shape[2]
                    + site1.shape[2] : site0.shape[2]
                    + site1.shape[2]
                    + site2.shape[2],
                    :,
                ],
            )

        else:
            assert np.array_equal(
                site0.data, site3.data[:, :, : site0.shape[2], : site0.shape[3]]
            )
            assert np.array_equal(
                site1.data,
                site3.data[
                    :,
                    :,
                    site0.shape[2] : site0.shape[2] + site1.shape[2],
                    site0.shape[3] : site0.shape[3] + site1.shape[3],
                ],
            )
            assert np.array_equal(
                site2.data,
                site3.data[
                    :,
                    :,
                    site0.shape[2]
                    + site1.shape[2] : site0.shape[2]
                    + site1.shape[2]
                    + site2.shape[2],
                    site0.shape[3]
                    + site1.shape[3] : site0.shape[3]
                    + site1.shape[3]
                    + site2.shape[3],
                ],
            )
"""

"""
def test_QTN_to_fMPO_and_back():
    fmpo = QTN_to_fMPO(mpo_classifier)
    QTN = fMPO_to_QTN(fmpo)

    # Check that conversion (and back) doesn't change data.
    for original, new in zip(mpo_classifier.tensors, QTN.tensors):
        assert np.array_equal(original.data, new.data)
"""


"""
def test_compress():

    # Truncated mpo needed otherwise dim(s) explodes when canonicalising
    truncated_mpo_classifier_data = [
        site.data[:, :1, :, :] if i < (n_sites - n_hairysites) else site.data
        for i, site in enumerate(mpo_classifier.tensors)
    ]
    truncated_mpo_classifier = data_to_QTN(truncated_mpo_classifier_data)
    truncated_mpo_classifier /= (
        truncated_mpo_classifier.H @ truncated_mpo_classifier
    ) ** 0.5

    compressed_mpo = compress_QTN(truncated_mpo_classifier, D=None, orthogonalise=False)

    # Check norm is still 1
    assert np.isclose((compressed_mpo.H @ compressed_mpo), 1)

    # Check overlap between initial classifier
    # and compressed mpo with D=None is 1.
    assert np.isclose((truncated_mpo_classifier.H @ compressed_mpo).norm(), 1)

    # Check canonicl form- compress procedure leaves mpo in mixed canonical form
    # center site is at left most hairest site.
    for n, site in enumerate(compressed_mpo.tensors):
        d, s, i, j = site.shape
        if n < (n_sites - n_hairysites):
            # reshape from (d, s, i, j) --> (d*j, s*i). As SVD was done like that.
            U = site.data.transpose(0, 3, 1, 2).reshape(d * j, s * i)
            Uh = U.conj().T
            assert np.array_equal(np.round(Uh @ U, 5), np.eye(s * i))
        else:
            # reshape from (d, s, i, j) --> (i, s*j*d). As SVD was done like that.
            U = site.data.transpose(2, 1, 3, 0).reshape(i, s * j * d)
            Uh = U.conj().T
            assert np.array_equal(np.round(U @ Uh, 5), np.eye(i))

    # Check compressed has right shape for range of different Ds
    for max_D in range(1, 5):
        compressed_mpo = compress_QTN(
            truncated_mpo_classifier, D=max_D, orthogonalise=False
        )
        for i, (site0, site1) in enumerate(
            zip(compressed_mpo.tensors, truncated_mpo_classifier)
        ):

            d0, s0, i0, j0 = site0.shape
            d1, s1, i1, j1 = site1.shape

            assert d0 == d1
            assert s0 == s1

            assert i0 <= max_D
            assert j0 <= max_D
    # TODO: Orthogonalisation test
"""
"""
def test_compress_one_site():

    one_site_mpo_train = mpo_encoding(
        mps_train, y_train, truncated_one_site_quimb_hairy_bitstrings
    )
    one_site_mpo_classifier = create_mpo_classifier(one_site_mpo_train, seed=420)

    one_site_compressed_mpo = compress_QTN(
        one_site_mpo_classifier, D=None, orthogonalise=False
    )

    # Check norm is still 1
    assert np.isclose((one_site_compressed_mpo.H @ one_site_compressed_mpo), 1)

    # Check overlap between initial classifier
    # and compressed mpo with D=None is 1.
    assert np.isclose((one_site_mpo_classifier.H @ one_site_compressed_mpo).norm(), 1)

    orthogonal_one_site_mpo = compress_QTN(
        one_site_mpo_classifier, D=None, orthogonalise=True
    )
    # Check Canonical form and orthogonal
    for k, site in enumerate(orthogonal_one_site_mpo.tensors):
        d, s, i, j = site.data.shape
        U = site.data.transpose(0, 2, 1, 3).reshape(d * i, s * j)
        Uh = U.conj().T
        if k < one_site_compressed_mpo.num_tensors - 1:
            assert np.isclose(Uh @ U, np.eye(s * j)).all()
        else:
            assert np.isclose(U @ Uh, np.eye(d * i)).all()
"""
"""
def test_batch_adding_mpos():
    mpos = []

    # Truncate images- required for Orthogonalisation procedure
    # to work.
    for mpo in mpo_train:
        truncated_mpo_data = [
            site.data[:, :1, :, :] if i < (n_sites - n_hairysites) else site.data
            for i, site in enumerate(mpo.tensors)
        ]
        truncated_mpo = data_to_QTN(truncated_mpo_data)
        truncated_mpo /= (truncated_mpo.H @ truncated_mpo) ** 0.5
        mpos.append(truncated_mpo)

    grouped_images = [
        [mpos[i] for i in range(n_train) if y_train[i] == label]
        for label in possible_labels
    ]
    grouped_mpos = [mpo_from_each_class for mpo_from_each_class in zip(*grouped_images)]

    batch_added_mpos = batch_adding_mpos(
        grouped_mpos, D=None, orthogonalise=False, compress=False
    )

    # Check, for compress = False that batch_adding_mpos is equivalent to simply adding all mpos.
    flattened_grouped_mpos = [item for sublist in grouped_mpos for item in sublist]
    added_mpos = flattened_grouped_mpos[0]
    for mpo in flattened_grouped_mpos[1:]:
        added_mpos = add_mpos(added_mpos, mpo)
    # "compression" required since compression done in batch addng changes the local values
    added_mpos = compress_QTN(added_mpos, D=None, orthogonalise=False)
    i = 0
    for site0, site1 in zip(batch_added_mpos.tensors, added_mpos.tensors):
        assert np.array_equal(site0.data, site1.data)

    # Check shape is correct for D != None.
    max_D = 5
    batch_added_mpos = batch_adding_mpos(
        grouped_mpos, D=max_D, orthogonalise=False, compress=True
    )
    for site0, site1 in zip(batch_added_mpos.tensors, mpos[0].tensors):
        d0, s0, i0, j0 = site0.shape
        d1, s1, i1, j1 = site1.shape

        assert d0 == d1
        assert s0 == s1

        assert i0 <= max_D
        assert j0 <= max_D
"""
